import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from ddpm_group import get_trainergroup
from utils import setup

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir 
    cfg.work_dir = work_dir

    port = int(np.random.randint(10000, 20000))
    try:
        mp.set_start_method("forkserver")
        mp.spawn(run_process, args=(cfg.ngpu, cfg, port), nprocs=cfg.ngpu, join=True)
    except Exception as e:
        print(e)


def run_process(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        tg = get_trainergroup(cfg, rank, init_test=True)
        tg.train()
    finally:
        destroy_process_group()



if __name__ == '__main__':
    main()