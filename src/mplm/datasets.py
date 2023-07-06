import dgl
import hydra.utils
import pandas as pd
import torch.nn.functional as F
import os
from random import randint

import torch.nn.functional as F
from transformers import AutoTokenizer

from copy import deepcopy
import numpy as np
from utils.data.preprocess import *
from utils.basics import *
from mplm.prompts import Demonstration

import utils.pkg.comm as comm
from dgl import node_subgraph, to_bidirected, remove_self_loop
import string
from bidict import bidict
from sklearn.model_selection import StratifiedShuffleSplit
import dgl
from langchain import PromptTemplate
from prompts import *
from mplm.prompts import Prompt
from dict2xml import dict2xml
from collections import Counter


def get_stratified_subset_split(labels, label_subset, valid_ids, n_split_samples):
    # Subset stratified split from all labels
    # valid_ids: available ids
    ids_left = valid_ids
    split_ids = {}
    for split, n_samples in n_split_samples.items():
        if n_samples > 0:
            split_ids[split] = np.random.permutation(np.concatenate(
                [ids_left[np.where(labels[ids_left] == l)[0][:n_samples]] for l in label_subset]))
            ids_left = np.setdiff1d(ids_left, split_ids[split])
        else:
            split_ids[split] = []
    return split_ids


def initialize_label_and_choices(all_label_info, label_subset=None):
    if label_subset is not None:
        label_info = all_label_info.iloc[label_subset].reset_index(drop=True)
    else:
        label_info = all_label_info
    label_info['choice'] = [string.ascii_uppercase[i] for i in range(len(label_info))]
    choice_to_label_name = bidict()
    choice_to_label_id = bidict()

    for i, row in label_info.iterrows():
        choice_to_label_name[row['choice']] = row['label_name']
        choice_to_label_id[row['choice']] = row['label_id']
    return label_info, choice_to_label_id, choice_to_label_name


def preprocess_ogb(ogb_name, process_mode, raw_data_path, raw_text_url, max_seq_len,
                   info_file, processed_text_file, n_nodes, n_labels, sample_per_class=1, demo: DictConfig = None,
                   label_text=None, subset_class=None, **kwargs):
    subset_class = subset_class or n_labels
    if subset_class != n_labels:
        label_subset = kwargs['_label_order'][:subset_class]
    else:
        label_subset = np.arange(n_labels)
    # ! Process Full Graph
    # Zhaocheng: please separate content in this if as a function preprocess()
    if comm.get_rank() == 0:
        g, labels, split_idx = load_ogb_graph_structure_only(ogb_name, raw_data_path)
        # Process and save supervision
        split_ids = {_: split_idx[_].numpy() for _ in ['train', 'valid', 'test']}
        if sample_per_class > 0 or subset_class != n_labels:
            # Top 5 frequently used classes are selected.
            g = to_bidirected(remove_self_loop(g))
            g, split_ids = subset_graph(g, sample_per_class, split_ids, labels, label_subset)
        g_info = EasyDict(splits=split_ids, labels=labels, n_nodes=g.num_nodes(),
                          IDs=np.arange(len(labels)))  # Default Graph Info for FULL graph
        if sample_per_class > 0 or subset_class != n_labels:
            g_info.IDs = g.ndata['_ID'].numpy()
            g_info.labels = g_info.labels[g_info.IDs]

            # Resplit according to few or one-shot.
            valid_ids = np.concatenate([v for k, v in split_ids.items()])
            n_train_samples = 1  # TODO to be revisited
            n_test_samples = round(sample_per_class * 0.8)  # To be defined in configs.
            n_split_samples = {'train': n_train_samples, 'test': n_test_samples,
                               'valid': sample_per_class - n_train_samples - n_test_samples}
            g_info.splits = get_stratified_subset_split(labels, label_subset, valid_ids, n_split_samples)
        pickle_save(g_info, info_file)
        del g

        if not os.path.exists(processed_text_file):
            if ogb_name == 'ogbn-arxiv':
                from utils.data.preprocess import process_raw_arxiv
                process_raw_arxiv(
                    labels, process_mode, ogb_name, raw_data_path, raw_text_url, max_seq_len,
                    processed_text_file, _label_info=kwargs['_label_info'])
            logger.info(f'Text preprocessing finished')

    comm.synchronize()
    # Load graph and text for every local-rank worker.
    g_info = pickle_load(info_file)
    g = load_ogb_graph_structure_only(ogb_name, raw_data_path)[0]
    # Zhaocheng: use uppercase G or graph for naming
    g = node_subgraph(g, g_info.IDs)
    g = to_bidirected(g)
    text, all_label_info = uf.pickle_load(processed_text_file)
    # self.text = full_data.iloc[g_info.IDs].reset_index(drop=True)

    split_ids = g_info.splits
    assert sum(labels == text.label_id) == n_nodes

    label_info, choice_to_label_id, choice_to_label_name = \
        initialize_label_and_choices(
            all_label_info, label_subset)
    text['gold_choice'] = text.apply(
        lambda x: label_info.choice.get(x['label_id'], 'Other Labels'),
        axis=1)
    text['pred_choice'] = np.nan

    label_lookup_funcs = (choice_to_label_id, choice_to_label_name)
    return g, g_info, text, label_info, label_lookup_funcs


def subset_graph(g, sample_per_class, split_ids, labels, label_subset, ensure_sub_label=False):
    # ! Subset labels first
    valid_ids = []
    for label in label_subset:
        subset_ids = np.where(labels == label)[0]
        subset_ids = np.intersect1d(subset_ids, th.where(g.in_degrees() > 0)[0].numpy())
        subset_ids = subset_ids[:sample_per_class] if sample_per_class else valid_ids
        valid_ids.append(subset_ids)
    # valid_ids = np.where(np.isin(labels, l-abel_subset))[0]
    valid_ids = np.concatenate(valid_ids)
    split_ids = {k: np.intersect1d(v, valid_ids) for k, v in split_ids.items()}

    # ! Subset graph
    if sample_per_class > 0 or label_subset != len(np.unique(labels)):
        subset_nodes = th.tensor(np.concatenate(list(split_ids.values())).astype(int))
        node_subset = sample_nodes(g, subset_nodes, [-1])[0]
        if ensure_sub_label:
            node_subset = np.intersect1d(node_subset, valid_ids)
        g = node_subgraph(g, node_subset)

    return g, split_ids


def preprocess_dgl(data_cfg: DictConfig):
    dataset = hydra.utils.instantiate(data_cfg['_init_args'])
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g, labels = dataset[0], dataset[0].ndata['label'].numpy()
    split_ids = EasyDict({s: np.random.permutation(np.where(g.ndata[f'{s}_mask'])[0])
                          for s in ['train', 'val', 'test']})

    g_info = EasyDict(splits=split_ids, labels=labels, n_nodes=g.num_nodes(),
                      IDs=np.arange(g.num_nodes()))
    # ! Get text attribute
    # Get label information
    all_label_info = pd.DataFrame.from_dict(
        {'label_id': [int(l) for l in data_cfg.label_name], 'label_name': data_cfg.label_name.values()})
    data = pd.DataFrame.from_dict({'label_id': labels})

    label_info, choice_to_label_id, choice_to_label_name = \
        initialize_label_and_choices(all_label_info, label_subset=None)
    data = pd.merge(data, label_info, how="left", on="label_id")
    data['text'] = data[data_cfg.text.mode]
    data['gold_choice'] = data.apply(
        lambda x: label_info.choice.get(x['label_id'], 'Other Labels'),
        axis=1)
    data['pred_choice'] = np.nan

    label_lookup_funcs = (choice_to_label_id, choice_to_label_name)
    return g, g_info, data, label_info, label_lookup_funcs


class TextualGraph(object):

    def __init__(self, cfg: DictConfig):  # Process split settings, e.g. -1/2 means first split
        self.cfg = cfg
        self.fanout = list(cfg.data.fanout)
        # ! Initialize Data Related
        if cfg.data.type == 'ogb':
            self.g, self.g_info, self.text, self.label_info, label_lookup_funcs = preprocess_ogb(
                **cfg.data)
            self.choice_to_label_id, self.choice_to_label_name = label_lookup_funcs
        elif cfg.data.type == 'dgl':
            self.g, self.g_info, self.text, self.label_info, label_lookup_funcs = preprocess_dgl(
                cfg.data)
            self.choice_to_label_id, self.choice_to_label_name = label_lookup_funcs
        self.n_labels = cfg.data.n_labels
        self.labels = self.g_info.labels
        # ! Initialize Prompt Related
        self.prompt = EasyDict()
        # Initialize classification prompt
        prompt_info = dict()
        assert (col := f"label_{cfg.data.label_text}") in self.label_info.columns, 'Unknown classification prompt mode.'
        prompt_info['label_description'] = label_text = '\n' + '\n'.join(
            f'{_["choice"]}: {_[col]}' for i, _ in self.label_info.iterrows()) + '\n'

        self.prompt.demo = Prompt(cfg.demo_cla, cfg.prompt_templates, **prompt_info)
        self.prompt.final_qa = Prompt(cfg.final_qa, cfg.prompt_templates, **prompt_info)
        self.prompt.cla = Prompt(cfg.cla, cfg.prompt_templates, **prompt_info)

        logger.info(self.prompt)
        # self.logger.info(f'mp_cfg template: {self.mp_cfg.template}')
        # !
        self.demo_select_func = cfg.demo.select_func
        self.demonstration = Demonstration(**cfg.demo, data=self.text, label_text=label_text,
                                           prompt=self.prompt.demo)

        self.split_ids = self.g_info.splits
        max_test_samples = min(len(self.split_ids.test), cfg.data.get('max_test_samples', 999999))
        if max_test_samples != len(self.split_ids.test):
            logger.warning(f'Evaluating on {max_test_samples}/{len(self.split_ids.test)} samples!')
            self.split_ids['test'] = self.split_ids['test'][:max_test_samples]
        self.all_ids = np.concatenate(list(self.g_info.splits.values()))
        val_count = self.text.iloc[self.all_ids].label_name.value_counts()
        logger.info(val_count / sum(val_count))
        split_len = {k: len(v) for k, v in self.split_ids.items()}
        # ! Build dataset default message
        self.build_input_msg()
        logger.info(f'Loaded dataset {cfg.data.name} with {split_len} and {self.g.num_edges()} edges')
        logger.info(f'Dataset initialization finished')

    def __getitem__(self, node_id):
        # ! Build Compute Graph
        if self.fanout is None:
            return self.text.iloc[node_id]
        node_list = [EasyDict(id=node_id, hop=0, neighbors=[])]
        pd.DataFrame.from_records(node_list)
        for k, fanout in enumerate(self.fanout):
            # sample k hop neighbors based on k-1 hop nodes
            k_hop_nodes = [n for n in node_list if n.hop == k]
            for node in k_hop_nodes:
                neighbors = self.g.in_edges(node.id)[0].cpu().numpy()
                node.neighbors = np.random.choice(neighbors, fanout) if fanout > 0 else neighbors
                node_list += [EasyDict(id=n, hop=k + 1, neighbors=[]) for n in neighbors]

        compute_graph = pd.DataFrame.from_records(node_list)
        compute_graph.set_index('id', inplace=True, drop=False)
        compute_graph['in_msg'] = self.text.iloc[compute_graph['id']]['in_msg']

        # ! Build Prompt Template
        demo_list = self.build_demo(node_id)
        demonstration = self.demonstration(demo_list)
        self.prompt.cla.update(demonstration=demonstration)
        return compute_graph

    def __len__(self):
        return len(self.all_ids)

    def build_input_msg(self, ):
        build_func = self.cfg.data.build_func
        if '-L' in build_func:  # Neighborhood Label Information
            # Parse Args
            hops, _, l_name_mode, conv_mode, visible_mode = [_ for _ in build_func.split('-')]
            hops = [int(_) for _ in hops]
            info = [dict() for _ in self.text.index]
            hop_to_str = {1: 'First-Order', 2: 'Second-Order', 3: 'Third-Order'}
            labels = self.g.ndata['label'].numpy()
            if visible_mode != 'All':
                labels[self.g.ndata['val_mask']] = -1
                labels[self.g.ndata['test_mask']] = -1
            for k in hops:
                k_hop_g = dgl.remove_self_loop(dgl.khop_graph(self.g, k))
                for _id, node in self.text.iterrows():
                    neighbors = k_hop_g.successors(_id)
                    nb_labels = self.g.ndata['label'][neighbors].numpy()
                    nb_labels = [self.label_info['choice'][_] for _ in nb_labels if _ != -1]
                    if l_name_mode == 'Name':
                        nb_labels = [f'{l}. {self.choice_to_label_name[l]}' for l in nb_labels]
                    if conv_mode == 'List':
                        info[_id][hop_to_str[k]] = str(nb_labels)
                    elif conv_mode == 'Count':
                        count = dict(Counter(nb_labels))
                        info[_id][hop_to_str[k]] = str(count)
                del k_hop_g
            self.text['in_msg'] = [dict2xml(d, wrap='information', indent='\t') for d in info]
        elif build_func == 'copy':
            self.text['in_msg'] = self.text['text']

    # @ Demonstration Related
    def build_demo(self, node_id):
        demo_ids = self.select_demo(node_id)
        demo_process = lambda x: {'information': x.in_msg, 'answer': x.gold_choice}
        demos = [demo_process(self.text.iloc[d_id]) for d_id in demo_ids]
        return demos

    def select_demo(self, node_id):
        if (n_demos := self.cfg.demo.n_samples) <= 0:
            return []
        one_fixed_sample_for_each_class_funcs = ['fixed-seed', 'class-prototype']
        if self.demo_select_func in one_fixed_sample_for_each_class_funcs:
            n_demo_per_class = max(n_demos // self.n_labels, 1)
            # Overwrite n_demos
            if not hasattr(self, 'demonstration_ids'):  # Initialize if haven't
                self.demonstration_ids = np.concatenate(
                    [self.split_ids.train[np.where(self.labels[self.split_ids.train] == l)[0][:n_demo_per_class]] for l
                     in
                     np.arange(self.n_labels)])
            return self.demonstration_ids
