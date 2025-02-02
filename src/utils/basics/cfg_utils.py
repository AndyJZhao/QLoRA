from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf

from .os_utils import *

UNIMPORTANT_CFG = EasyDict(
    fields=['gpus', 'debug', 'wandb', 'env', 'uid',
            'local_rank', 'cmd', 'file_prefix'],
    prefix=['_'],
    postfix=['_path', '_file', '_dir']
)


def cfg_dict(cfg):
    if isinstance(cfg, DictConfig):
        return EasyDict(OmegaConf.to_object(cfg))
    elif isinstance(cfg, dict):
        return cfg
    else:
        raise ValueError(f'Unsupported config type for {type(cfg)}')


def cfg_to_file_name(cfg, compress=False):
    """To avoid conflict while launching multiple parallel runs together,
    here we map the config to a unique string file name.
    """
    MAX_FNAME = 255

    def _cfg_to_str(cfg):
        if isinstance(cfg, dict):
            if '_file_' in cfg:
                return cfg['_file_']
            return ''.join([_cfg_to_str(_) for _ in cfg.values()])
        else:
            return str(cfg)

    _cfg = cfg_dict(cfg) if isinstance(cfg, DictConfig) else cfg
    imp_cfg = get_important_cfg(cfg)
    s = f"{_cfg_to_str(imp_cfg)}"

    for _ in ['/', ' ', ':', 'class', 'None', '~', 'yaml',
              '\'', '[', ']', '(', ')', '{', '}', '.', ',']:
        s = s.replace(_, '')

    for k, v in {'True': 'T', 'False': 'F'}.items():
        s = s.replace(k, v)

    if compress:
        _map = lambda list_of_ascii: 61 + sum(list_of_ascii) % 29
        if len(s) > MAX_FNAME:
            # Post-processing to map to length <255
            # map to devision compression method, the prime is selected as 29
            ascii_code = np.array(list(s.encode('ascii')))
            compressed_ascii = [_map(_) for _ in np.array_split(ascii_code, MAX_FNAME)]
            s = ''.join(chr(_) for _ in compressed_ascii)
    else:
        s = '/'.join([s[i:i + 255] for i in range(0, len(s), 255)])
    return s


# ! Get config

def add_cmd_line_args_to_hydra_cfg(cfg: DictConfig):
    # Previously, we need to access choices, e.g. exp, in default list
    # Abandoned. Using ${hydra:runtime.choices.exp} instead
    OmegaConf.set_struct(cfg, False)
    cmd_arg = {}
    for _ in sys.argv:
        if '.py' not in _:
            sep = '=' if '=' in _ else ' '
            k, v = _.split(sep)
            cmd_arg[k] = v
    cfg.cmd = cmd_arg


def get_important_cfg(cfg, reserve_file_cfg=True, unimportant_cfg=UNIMPORTANT_CFG):
    uimp_cfg = cfg.get('_unimportant_cfg', unimportant_cfg)
    imp_cfg = OmegaConf.to_object(cfg)

    def is_preserve(k: str):
        judge_file_setting = k == '_file_' and reserve_file_cfg
        prefix_allowed = (not any([k.startswith(_) for _ in uimp_cfg.prefix])) or judge_file_setting
        postfix_allowed = not any([k.endswith(_) for _ in uimp_cfg.postfix])
        field_allowed = k not in uimp_cfg.fields
        return prefix_allowed and postfix_allowed and field_allowed

    imp_cfg = subset_dict_by_condition(imp_cfg, is_preserve)
    return imp_cfg


def print_important_cfg(cfg, log_func=logger.info):
    log_func(OmegaConf.to_yaml(get_important_cfg(cfg, reserve_file_cfg=False)))


# ! Resolvers

def calc_bsz_per_dev(eq_bsz, gpus):
    return eq_bsz // len(gpus)


def ternary_operator(condition, val_if_true, val_if_false):
    return val_if_true if condition else val_if_false


def calc_bsz_and_grad_acc_per_dev(eq_batch_size, max_bsz_dict, sv_info, min_bsz=2):
    def get_max_batch_size(gpu_mem, max_bsz_dict):
        quantized_gpu_mem = floor_quantize(gpu_mem, max_bsz_dict.keys())
        return max_bsz_dict[quantized_gpu_mem]

    max_bsz_per_gpu = get_max_batch_size(sv_info.gpu_mem, max_bsz_dict)
    gpus = os.environ['CUDA_VISIBLE_DEVICES']
    n_gpus = len(gpus.split(',')) if gpus != '' else 1
    logger.info(f'N-GPUs={n_gpus}')

    def find_grad_acc_steps(bsz_per_gpu):
        # Find batch_size and grad_acc_steps combination that are DIVISIBLE!
        grad_acc_steps = eq_batch_size / bsz_per_gpu / n_gpus
        if grad_acc_steps.is_integer():
            return bsz_per_gpu, int(grad_acc_steps)
        elif grad_acc_steps:
            if bsz_per_gpu >= min_bsz:
                return find_grad_acc_steps(bsz_per_gpu - 1)
            else:
                raise ValueError(
                    f'Cannot find grad_acc_step with integer batch_size greater than {min_bsz}, '
                    f'eq_bsz={eq_batch_size}, n_gpus={n_gpus}')

    batch_size, grad_acc_steps = find_grad_acc_steps(max_bsz_per_gpu)
    logger.info(
        f'Eq_batch_size = {eq_batch_size}, bsz={batch_size}, grad_acc_steps={grad_acc_steps}, ngpus={n_gpus}')
    return batch_size, grad_acc_steps


# Register resolvers
OmegaConf.register_new_resolver('calc_bsz_per_dev', calc_bsz_per_dev)
OmegaConf.register_new_resolver('condition', ternary_operator)
