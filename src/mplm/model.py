import dgl
import pandas as pd
from langchain import PromptTemplate
from sklearn.metrics import classification_report

from llm.llm import LLM
from mplm.datasets import TextualGraph
from utils.basics import *


class MessagePassingLM(object):
    def __init__(self, cfg: DictConfig, data: TextualGraph, llm: LLM, _logger,
                 save_file=None, cla: DictConfig = None, msg: DictConfig = None,
                 agg: DictConfig = None, comb: DictConfig = None,
                 max_new_tokens=20, gen_mode='choice', **kwargs):
        # Zhaocheng: list is mutable and should never be used as default arguments
        # Use tuple instead
        # fanout -> num_neighbor_per_hop
        self.cfg = cfg
        self.gen_mode = gen_mode
        self.cla_cfg, self.msg_cfg, self.agg_cfg, self.comb_cfg = cla, msg, agg, comb
        self.data = data
        self.g, self.g_info, self.text = data.g, data.g_info, data.text
        self.llm = llm
        self.g = dgl.to_bidirected(self.g)
        self.hop = len(data.fanout)
        self.save_file = save_file
        self.logger = _logger
        self.max_new_tokens = max_new_tokens
        # ! Classification prompt

        # self.prompt.msg = prompt_init(msg['_prompt_args'])
        # self.prompt.agg = prompt_init(agg['_prompt_args'])

        self.text['out_msg'] = 'NA'
        self.text['conversation'] = 'NA'
        self.text['generated_text'] = 'NA'
        # leave 10 for generation
        # self.cla_msg_cut_off = cfg.llm.max_seq_len - len((self.prompt.cla)) - 10

    def message_passing(self, compute_graph, query_node, proxy_node):
        # Standard GNNs performs message passing in a two-step manner where
        # the neighborhood information is firstly transformed by a Message function and
        # then aggregated by an Aggregate function.
        # In MPLM, we add another step "Combine" to combine the center node information to the
        # final message.
        def run_prompt(prompt_template: PromptTemplate, prompt_args=None, max_new_tokens=20):
            input_text = prompt_template.format(
                **{k: v for k, v in prompt_args.items() if k in prompt_template.input_variables})
            return self.llm.generate_text(input_text, max_new_tokens)

        # ! Step 1 Message: transform message of EACH neighbor [n_nb, msg_out_len] -> [n_nb , msg_hid_len]
        if self.msg_cfg.mode == 'copy':
            neighborhood_msg = [f'\'{compute_graph.loc[_, "msg"]}\''
                                for _ in proxy_node.neighbors]
        else:
            neighborhood_msg = [
                run_prompt(self.prompt.msg, prompt_args=EasyDict(
                    query_node_msg=query_node.msg,
                    proxy_node_hop=proxy_node.hop,
                    proxy_node_msg=proxy_node.msg,
                    neighborhood_msg=compute_graph.loc[n].msg,
                ),
                           max_new_tokens=self.msg_cfg.max_tokens)
                for n in proxy_node.neighbors]

        # ! Step 2 Aggregate: combine all neighborhood to one msg [n_nb , msg_max_tokens] -> [nb_msg_len]
        if self.comb_cfg.mode == 'concat':
            neighborhood_msg = ';'.join(neighborhood_msg)
            logger.debug(neighborhood_msg)
        else:
            raise NotImplementedError

        # ! Step 3 Combine: fuse obtained neighborhood msg and self-msg to one final msg

        if self.comb_cfg.mode == 'self+neighbor':
            return proxy_node.msg + f'\nNeighborhood information: "{neighborhood_msg}"'
        elif self.comb_cfg.mode == 'neighbor_only':
            return f'\nNeighborhood information: "{neighborhood_msg}"'
        else:
            raise NotImplementedError

    def classification(self, information):
        qa_instruction = self.data.prompt.final_qa(information=information)
        prompt = self.data.prompt.cla(qa_instruction=qa_instruction)
        prompt = prompt + ' ' if prompt.endswith(':') else prompt  # ! Critical
        # prompt = f"Question: What's the topic of academic paper [{query_node.msg}]?\nA: Computer Vision and Pattern
        # Recognition\nB: Machine Learning\nC: Information Theory\nD: Computation and Language\n\nAnswer: "
        if self.gen_mode == 'choice':
            generated = self.llm.generate_text(prompt, max_new_tokens=1, choice_only=True)
            pred_choice = generated[-1] if len(generated) > 0 else 'NULL'
        else:
            generated = self.llm.generate_text(prompt, self.max_new_tokens)
            pred_choice = generated.split('<answer>')[-1][0]
        output = {'conversation': prompt + generated,
                  'pred_choice': pred_choice,
                  'generated_text': generated,
                  }
        return output

    def eval_and_save(self, step, node_id):
        res_df = self.text.dropna()
        res_df['correctness'] = res_df.apply(lambda x: x['gold_choice'] == x['pred_choice'], axis=1)
        res_df.to_csv(self.save_file)
        logger.info(f'Saved results to {self.save_file}')
        acc = res_df['correctness'].mean()
        try:
            valid_choice_rate = res_df['pred_choice'].isin(self.data.choice_to_label_id.keys()).mean()
            acc_in_valid_choice = acc / valid_choice_rate
            result = {'acc': acc, 'valid_choice_rate': valid_choice_rate, 'acc_in_valid_choice': acc_in_valid_choice}

            valid_df = res_df[res_df['pred_choice'].isin(self.data.choice_to_label_id.keys())]
            valid_df['true_choices'] = valid_df.apply(lambda x: self.data.label_info.choice[x['label_id']], axis=1)
            result.update({
                f'PD/{choice}.{self.data.choice_to_label_name[choice]}': cnt / len(valid_df)
                for choice, cnt in
                valid_df.pred_choice.value_counts().to_dict().items()})
            sample = {f'sample_{k}': v
                      for k, v in self.data.text.iloc[node_id].to_dict().items()}
            self.logger.info(sample)
            self.logger.wandb_log({**result, 'step': step})
        except Exception as e:
            self.logger.critical(f'First sample {res_df.iloc[0]}')
            raise ValueError(f'An error occurred while summarizing experiments, {e}')

        #  ! Save statistics to results
        # y_true, y_pred = [valid_df.apply(lambda x: self.data.l_choice_to_id[x[col]], axis=1).tolist() for col in
        #                   ('true_choices', 'pred_choice')]
        # result['cla_report'] = classification_report(
        #     y_true, y_pred, output_dict=True,
        #     target_names=self.data.label_info.label_name.tolist())
        result['out_file'] = self.save_file
        self.logger.wandb_summary_update({**result, **sample})
        self.logger.info(result)
        return result

    def __call__(self, query_node_id, log_sample=False):
        # ! Build computation graph
        compute_graph = self.data[query_node_id]
        query_node = compute_graph.loc[query_node_id]

        # ! Message Passing: Map NeighborInfo and reduce
        # for hop in reversed(range(self.hop)):
        #     for _, node in compute_graph.query(f'hop=={hop}').iterrows():
        #         if len(node.neighbors) > 0 and self.comb_cfg.mode != 'self_only':
        #             compute_graph.at[node.id, 'msg'] = self.message_passing(compute_graph, query_node, node)
        compute_graph['out_msg'] = compute_graph['in_msg']
        # ! Classification
        output = self.classification(information=compute_graph.loc[query_node_id].out_msg)

        self.text.loc[query_node.id, 'pred_choice'] = output['pred_choice']
        self.text.loc[query_node.id, 'out_msg'] = compute_graph.at[query_node.id, 'out_msg']
        self.text.loc[query_node.id, 'conversation'] = output['conversation']
        self.text.loc[query_node.id, 'generated_text'] = output['generated_text']
