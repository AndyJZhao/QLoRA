import wandb
import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install
from rich import print

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=True,
        console=Console(width=165),
        enable_link_path=False
    )],
)
# Default logger
logger = rich_logger = logging.getLogger("rich")
install(show_locals=True, width=150)
print("Rich Logger and Traceback initialized.")


class WandbLogger:
    def __init__(self, wandb_settings, local_rank=0, level='info'):
        self.wandb = wandb_settings
        self.wandb_on = wandb_settings.id is not None
        self.local_rank = local_rank
        self.logger = rich_logger  # Rich logger
        self.logger.setLevel(getattr(logging, level.upper()))
        self.info = self.logger.info
        self.critical = self.logger.critical
        self.debug = self.logger.debug
        self.info = self.logger.info

    def log(self, *args, level='', **kwargs):
        if self.local_rank <= 0:
            self.logger.log(getattr(logging, level.upper()), *args, **kwargs)

    def log_fig(self, fig_name, fig_file):
        if self.wandb_on and self.local_rank <= 0:
            wandb.log({fig_name: wandb.Image(fig_file)})
        else:
            self.log('Figure not logged to Wandb since Wandb is off.', 'ERROR')

    def wandb_log(self, wandb_dict, level='info'):
        if self.wandb_on and self.local_rank <= 0:
            wandb.log(wandb_dict)
        self.log(wandb_dict, level=level)

    def wandb_summary_update(self, result):
        if self.wandb_on and self.local_rank <= 0:
            wandb.summary.update(result)

    def metric_log(self, log_dict, wandb_dict=None, level='WARNING'):
        self.wandb_log(log_dict if wandb_dict is None else wandb_dict)
        round_float = lambda x: f'{x:.4f}' if x > 1e-4 else x
        map_funcs = {int: lambda x: f'{x:03d}', float: round_float}
        log_map = lambda v: map_funcs[type(v)](v) if type(v) in map_funcs else v
        log_dict.update({k: log_map(v) for k, v in log_dict.items()})
        # log_dict = {k: v for k, v in log_dict.items() if k[0] != '_'}
        # self.log(' | '.join([f'{k} {v}' for k, v in log_dict.items()]), level)


def wandb_finish(result=None):
    if wandb.run is not None:
        wandb.summary.update(result or {})
        wandb.finish()
