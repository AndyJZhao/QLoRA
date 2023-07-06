from omegaconf import OmegaConf, DictConfig
from pandas import DataFrame
import numpy as np
from string import Formatter
import xml.etree.ElementTree as ET


def get_string_args(s):
    return [fn for _, fn, _, _ in Formatter().parse(s) if fn is not None]


def preprocess_prompt_config(prompt_cfg, lookup_dict):
    cfg_dict = OmegaConf.to_object(prompt_cfg)
    processed_dict = {}
    for k, v in cfg_dict.items():
        if not k.startswith('_') and k in lookup_dict:  # Is a prompt template
            retrieved_str = lookup_dict[k][v]
            processed_dict[k] = preprocess_yaml_fstring(retrieved_str)
    return processed_dict


def preprocess_yaml_fstring(s):
    s = s.replace('\n', '')
    s = s.replace('\\n ', '\n')
    s = s.replace('\\n', '\n')
    return s


class Prompt:
    # A simpler class than Langchain.PromptTemplate
    # With some tweaks
    def __init__(self, prompt_cfg: DictConfig, template_lookup_dict, **kwargs):
        cfg_dict = preprocess_prompt_config(prompt_cfg, template_lookup_dict)
        self.template = cfg_dict.pop('prompt')
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}
        self._variables = {**cfg_dict, **kwargs}
        self._var_set = set(get_string_args(self.template))

    def update(self, **kwargs):
        self._variables.update(kwargs)

    def __call__(self, assert_vars=True, **kwargs):
        args = {**self._variables, **kwargs}
        if assert_vars:
            assert len(set(args.keys()) - self._var_set) >= 0, f'{self._var_set - set(args.keys())} not given.'
        return self.template.format(**args)

    def __str__(self):
        # For unknown args, use {arg} instead.
        kwargs = {k: self._variables.get(k, '{' + k + '}') for k in self._var_set}
        return self.__call__(**kwargs)

    def __repr__(self):
        return f'PromptTemplate: <<{self.__str__()}>>'


class Demonstration:
    def __init__(self, data: DataFrame, prompt: Prompt, n_separators=2, keep_label_description=False, label_text='',
                 prefix='\n\nHere are a few examples: \n\n', suffix='\n\n', **kwargs):
        self.data = data
        self.prefix = prefix
        self.suffix = suffix
        self.sep = '\n' * n_separators
        self.prompt = prompt
        self.label_description = label_text if keep_label_description else ''

    def __call__(self, demo_list, **kwargs):
        if len(demo_list) == 0:
            return ''
        else:
            demonstration = self.prefix + self.sep.join(
                self.prompt(**d, label_description=self.label_description) for d in demo_list) + self.suffix
            return demonstration
