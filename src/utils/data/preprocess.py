import time
from copy import deepcopy

import dgl
import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm
from transformers import AutoTokenizer

import utils.basics as uf
from utils.pkg.dgl_utils import sample_nodes
import os
from easydict import EasyDict
import utils.pkg.comm as comm


def _subset_graph(g, cf, sup):
    splits = ['train', 'valid', 'test']
    if cf.data.sample_per_class > 0:
        subset = lambda x: x[:round(len(x) * cf.data.sample_per_class)].tolist()
        split_ids = {_: subset(sup[_]) for _ in splits}
        subset_nodes = th.tensor(sum([split_ids[_] for _ in splits], []))
        node_subset = sample_nodes(g, subset_nodes, [-1])[0]
        # g.ndata['labels'] = th.from_numpy(sup['labels'])
        g = dgl.node_subgraph(g, node_subset)
        # new_split_ids = {_: [] for _ in splits}
        # for i in range(g.num_nodes()):
        #     get_split=lambda _: g.ndata['_ID'][i] in split_ids[_ ]
        #     for split in splits:
        #         if get_split(split):
        #             new_split_ids[split].append(i)
        # split_ids = {f'{_}': new_split_ids[_] for _ in splits}
    else:
        split_ids = {f'{_}': sup[_] for _ in splits}
    split_len = {_: len(split_ids[f'{_}']) for _ in splits}
    log_func = cf.logger if hasattr(cf, 'log') else print
    log_func(f'Loaded dataset {cf.dataset} with {split_len} and {g.num_edges()} edges')
    return g, split_ids


def plot_length_distribution(node_text, tokenizer, g):
    sampled_ids = np.random.permutation(g.nodes())[:10000]
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    tokenized = tokenizer(get_text(sampled_ids), padding='do_not_pad').data['input_ids']
    node_text['text_length'] = node_text.apply(lambda x: len(x['text'].split(' ')), axis=1)
    pd.Series([len(_) for _ in tokenized]).hist(bins=20)
    import matplotlib.pyplot as plt
    plt.show()


def process_graph_structure(g, cf):
    a = time.time()
    g = dgl.to_bidirected(g)
    g_info = cf.data.gi
    if cf.data.sample_per_class > 0:
        g = dgl.node_subgraph(g, cf.data.gi.IDs)
        g = g.remove_self_loop().add_self_loop()
    if cf.model in {'RevGAT', 'GCN'}:
        # add self-loop
        print(f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")
    print(f'process_graph OK!, spend {time.time() - a}')
    if 'ind' in cf.dataset:
        visible_nodes = list(set(g.nodes().numpy().tolist()) - set(g_info.splits['test']))
        g = dgl.node_subgraph(g, visible_nodes)
    if 'IND' in cf.dataset or 'Ind' in cf.dataset:
        test_ids = g_info.splits['test']
        edges_to_rm = th.cat((g.in_edges(test_ids, form='eid'), g.out_edges(test_ids, form='eid')))
        g = dgl.remove_edges(g, edges_to_rm)
        g = g.remove_self_loop().add_self_loop()
    return g  # .to(cf.device)


def process_pyg_graph_structure(data, cf):
    from torch_sparse import SparseTensor
    from torch_geometric.utils import to_undirected

    path = '../adj_gcn.pt'
    a = time.time()
    if os.path.exists(path):
        adj = th.load(path)
    else:
        N = data.num_nodes
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        row, col = data.edge_index
        print('Computing adj...')
        adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        adj = adj.set_diag()
        deg = adj.sum(dim=1).to(th.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        th.save(adj, path)

    adj = adj.to_scipy(layout='csr')
    print(f'process_graph OK!, spend {time.time() - a}')
    del data
    return adj


def load_ogb_graph_structure_only(ogb_name, raw_data_path):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(ogb_name, root=uf.init_path(raw_data_path))
    g, labels = data[0]
    split_idx = data.get_idx_split()
    labels = labels.squeeze().numpy()
    return g, labels, split_idx


def load_pyg_graph_structure_only(cf):
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(cf.data.ogb_name, root=uf.init_path(cf.data.raw_data_path))
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    return data, split_idx


def load_graph_info(cf):
    d = cf.data
    # ! Process Full Graph
    if not d.is_processed('g_info'):
        # Load OGB
        if comm.get_rank() == 0:
            g, labels, split_idx = load_ogb_graph_structure_only(cf)
            # Process and save supervision
            splits = {**{f'{_}': split_idx[_].numpy() for _ in ['train', 'valid', 'test']}, 'labels': labels}
            g = dgl.to_bidirected(g)
            g, splits = _subset_graph(g, cf, splits)
            g_info = EasyDict(splits=splits, labels=labels, n_nodes=g.num_nodes())
            if d.sample_per_class > 0:
                g_info.IDs = g.ndata['_ID'].numpy()
                g_info.labels = g_info.labels[g_info.IDs]
                g_info.val_test = g_info.val_test[g_info.IDs]
            d.save_g_info(g_info)
            del g
        else:
            # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
            print(f'Waiting for feature processing on LOCAL_RANK #{cf.local_rank}')
            while not d.is_processed('g_info'):
                time.sleep(2)  # Check if processed every 2 seconds
            print(f'Detected processed feature, LOCAL_RANK #{cf.local_rank} start loading!')
            time.sleep(5)  # Wait f
    g_info = uf.pickle_load(d._g_info_file)
    return g_info


def process_raw_arxiv(labels, mode, ogb_name, raw_data_path, raw_text_url, max_seq_len,
                      processed_text_file, chunk_size=50000, _label_info=None, **kwargs):
    def merge_by_ids(meta_data, node_ids, label_info):
        meta_data.columns = ['node_id', "Title", "Abstract"]
        # meta_data.drop([0, meta_data.shape[0] - 1], axis=0, inplace=True)  # Drop first and last in Arxiv full
        # dataset processing
        meta_data['node_id'] = meta_data['node_id'].astype(np.int64)
        meta_data.columns = ["mag_id", "title", "abstract"]
        data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
        data = pd.merge(data, label_info, how="left", on="label_id")
        return data

    def read_ids_and_labels():
        _ = f'{raw_data_path}{ogb_name.replace("-", "_")}/mapping/'
        category_path_csv = f"{_}labelidx2arxivcategeory.csv.gz"
        paper_id_path_csv = f"{_}nodeidx2paperid.csv.gz"  #
        paper_ids = pd.read_csv(paper_id_path_csv)
        label_info = pd.read_csv(category_path_csv)
        paper_ids.columns = ['node_id', "mag_id"]
        label_info.columns = ["label_id", "label_raw_name"]
        paper_ids["label_id"] = labels[paper_ids['node_id']]
        label_info['label_raw_name'] = label_info.apply(lambda x: x['label_raw_name'].split('arxiv cs ')[1].upper(),
                                                        axis=1)
        label_info['label_name'] = label_info.apply(lambda x: _label_info[x['label_raw_name']].split(' - ')[0], axis=1)
        label_info['label_alias'] = label_info.apply(lambda x: f"cs.{x['label_raw_name']}", axis=1)
        label_info['label_alias+name'] = label_info.apply(lambda x: f"{x['label_alias']} ({x['label_name']})", axis=1)
        label_info['label_description'] = label_info.apply(lambda x: _label_info[x['label_raw_name']], axis=1)
        return label_info, paper_ids  # 返回类别和论文ID

    def process_raw_text_df(meta_data, node_ids, label_info):
        data = merge_by_ids(meta_data.dropna(), node_ids, label_info)
        data = data[~data['title'].isnull()]
        text_func = {
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
            'T': lambda x: x['title'],
        }
        # Merge title and abstract
        data['text'] = data.apply(text_func[mode], axis=1)
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:max_seq_len]), axis=1)
        return data

    from ogb.utils.url import download_url
    # Get Raw text path
    print(f'Loading raw text for {ogb_name}')
    raw_text_path = download_url(raw_text_url, raw_data_path)

    label_info, node_ids = read_ids_and_labels()
    df_list = []
    for meta_data in tqdm(pd.read_table(raw_text_path, header=None, chunksize=chunk_size, skiprows=[0])):
        # Load part of the dataframe to prevent OOM.
        df_list.append(process_raw_text_df(meta_data, node_ids, label_info))
    processed_df = pd.concat(df_list).sort_index()
    assert sum(processed_df.node_id == np.arange(len(labels))) == len(labels)
    uf.pickle_save((processed_df, label_info), processed_text_file)
    return processed_df
