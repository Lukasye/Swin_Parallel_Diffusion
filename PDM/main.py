import hydra
from hydra.core.hydra_config import HydraConfig

from ddpm_group import Swin_TrainerGroup
from utils import get_device

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir 
    cfg.work_dir = work_dir

    tg = Swin_TrainerGroup(cfg, rank=0)
    tg.train_sr()


if __name__ == '__main__':
    main()