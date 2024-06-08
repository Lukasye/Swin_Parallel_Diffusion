import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np

from ddpm_group import Swin_TrainerGroup, DWT_TrainerGroup
import metrics.fid_npzs as fid_npzs
from metrics import dnnlib
from metrics.sr_metrics import Measure
import utils


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir 
    cfg.work_dir = os.path.join(os.getcwd(), work_dir)
    for_swin(cfg)
    # for_dwt(cfg)


def for_dwt(cfg):
    checkpoint_path = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(shifting step 4)/checkpoint_meta/checkpoint_meta.pth'
    tg = DWT_TrainerGroup(cfg, rank=-1, init_test=False, train=True)
    tg.load_checkpoints(checkpoint_path)
    sample = next(iter(tg.sample_dataloader))
    tg.sample_v2(sample)


def for_swin(cfg):
    checkpoint_path = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/2024.06.07/checkpoint_meta/checkpoint_meta.pth'
    tg = Swin_TrainerGroup(cfg, rank=0, init_test=False, train=True)
    tg.load_checkpoints(checkpoint_path)
    num_samples = cfg.evaluation.num_samples
    fid = cfg.evaluation.fid
    sr = cfg.evaluation.sr
    # tg.load_checkpoints(checkpoint_path)
    
    save_path = tg.paths['fid']
    measure = Measure(cfg.training.scaling_factor)

    num_iter = 20
    print(f'Generating samples...')

    counter = 0
    for batch in tg.sample_dataloader:
        batch = tg.all_to_device(batch)
        start = time.time()
        result = tg.sample(batch, save=False)
        end = time.time()
        duration = end - start
        print(f'Duration: {duration}')
        utils.save_images(result, save_path=os.path.join(save_path, f'sample_{counter}.jpg'), unnormalized=True)

        if sr:
            print('Analyzing SR Metrics...')
            # for b in range(result.shape[0]):
            #     measure.measure(result[b], batch[0][b], batch[1][b], tg.scaling_factor)
            measure.measure_batch(result, batch)
            print(measure.get_result())

        if counter > num_iter:
            break
        counter += 1

    print(measure.get_result())

    if fid:
        ref_path = cfg.evaluation.ref

        print(f'Loading dataset reference statistics from "{ref_path}"...')
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))
        mu, sigma = fid_npzs.calculate_inception_stats_npz(image_path=save_path, num_samples=num_samples, device=tg.device)
        print('Calculating FID...')
        fid = fid_npzs.calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'{save_path.split("/")[-1]}, {fid:g}')


if __name__ == '__main__':
    main()