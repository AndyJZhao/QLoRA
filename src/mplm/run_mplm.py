import os
import sys

# Zhaocheng: this is because you use weird relative import in some files
# e.g. [llm/llama.py] from llm.llm import LLM
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
os.chdir(root_path)
sys.path.append(root_path + 'src')

from utils.basics import init_env_variables, time_logger, DictConfig, wandb_finish, remove_file_or_path

init_env_variables()

import hydra

from mplm.model import MessagePassingLM
from mplm.datasets import TextualGraph
from utils.project.exp import init_experiment
from llm import CpuFakeDebugLLM
from tqdm import tqdm
import torch


@time_logger
@hydra.main(config_path='../../configs', config_name='main_cfg', version_base=None)
def train_mplm(cfg: DictConfig):
    cfg, logger = init_experiment(cfg)
    data = TextualGraph(cfg=cfg)
    # llm = hydra.utils.instantiate(cfg.llm)
    use_fake_llm = (not torch.cuda.is_available()) and 'gpt' not in cfg.llm.name
    if use_fake_llm:
        llm = CpuFakeDebugLLM()  # Use local CPU for faster debugging
    else:
        llm = hydra.utils.instantiate(cfg.llm, data=data)

    model = MessagePassingLM(cfg, data, llm, logger, **cfg.model)
    for i, node_id in tqdm(enumerate(data.split_ids.test), 'Evaluating...'):
        # for i, node_id in track(enumerate(data.split_ids.test[:10]), 'Evaluating...'):
        is_evaluate = i % cfg.eval_freq == 0 and i != 0
        model(node_id, log_sample=is_evaluate)
        if is_evaluate:
            model.eval_and_save(i, node_id)

    result = model.eval_and_save(i, node_id)
    logger.info('Training finished')
    wandb_finish(result)

    # ! Remove temp files if specified
    if cfg.get('remove_temp', False):
        remove_file_or_path(cfg.working_dir)


if __name__ == "__main__":
    train_mplm()
