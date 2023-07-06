import torch
from torch import distributed as dist
import utils.pkg.comm as comm
from uuid import uuid4

from utils.basics import *

proj_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
PROJ_CONFIG_FILE = 'config/proj.yaml'


def set_seed(seed):
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed + comm.get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_init(gpus):
    import torch as th
    device = th.device('cpu')
    if gpus != '-1' and th.cuda.is_available():  # GPU
        if comm.get_rank() >= 0:  # DDP
            th.cuda.set_device(comm.get_rank())
            device = th.device(comm.get_rank())
        else:  # Single GPU
            device = th.device("cuda:0")
    return device


def generate_unique_id(cfg):
    """Generate a Unique ID (UID) for (1) File system (2) Communication between submodules
    By default, we use time and UUID4 as UID. UIDs could be overwritten by wandb or UID specification.
    """
    #
    if cfg.get('uid') is not None and cfg.wandb.id is not None:
        assert cfg.get('uid') == cfg.wandb.id, 'Confliction: Wandb and uid mismatch!'
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.wandb.id or cfg.get('uid')
    uid = given_uid if given_uid else cur_time + str(uuid4()).split('-')[0]
    return uid


def init_experiment(cfg):
    # Prevent ConfigKeyError when accessing non-existing keys
    OmegaConf.set_struct(cfg, False)
    wandb_init(cfg)
    set_seed(cfg.seed)
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        # comm.init_process_group("nccl", init_method="proj://")
        comm.init_process_group("nccl", init_method="env://")

    # In mplm working directory is initialized by mplm and shared by LM and GNN submodules.
    cfg.uid = generate_unique_id(cfg)
    cfg.working_dir = init_path(cfg.working_dir)
    init_path(cfg.out_file_prefix)
    cfg.local_rank = comm.get_rank()
    _logger = WandbLogger(cfg.wandb, level='info', local_rank=cfg.local_rank)
    _logger.info(f'Local_rank={cfg.local_rank}, working_dir = {cfg.working_dir}')
    print_important_cfg(cfg, _logger.info)
    return cfg, _logger


def wandb_init(cfg) -> None:
    os.environ["WANDB_WATCH"] = "false"
    if cfg.get('use_wandb', False) and comm.get_rank() <= 0:
        try:
            WANDB_API_KEY, WANDB_DIR, WANDB_PROJ, WANDB_ENTITY = (
                ENV_VARS[k.lower()] for k in ['WANDB_API_KEY', 'WANDB_DIR', 'WANDB_PROJ',
                                              'WANDB_ENTITY'])
            os.environ['WANDB_API_KEY'] = WANDB_API_KEY
            wandb_dir = os.path.join(proj_path, WANDB_DIR)

            # ! Create wandb session
            if cfg.wandb.id is None:
                # First time running, create new wandb
                wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, dir=wandb_dir,
                           reinit=True, config=get_important_cfg(cfg), name=cfg.wandb.name)
            else:
                print(f'Resume from previous wandb run {cfg.wandb.id}')
                wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, reinit=True,
                           resume='must', id=cfg.wandb.id)
            cfg.wandb.id, cfg.wandb.name, cfg.wandb.sweep_id = wandb.run.id, wandb.run.name, wandb.run.sweep_id
            return
        except:
            logger.info('WANDB NOT INITIALIZED.')
    os.environ["WANDB_DISABLED"] = "true"
    return
