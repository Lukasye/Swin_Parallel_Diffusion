import os
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import torchvision
from tqdm import tqdm
import numpy as np

from ddpm_group import get_trainergroup, Swin_TrainerGroup
from utils import get_device
import fid_npzs
import dnnlib
import utils


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir 
    cfg.work_dir = work_dir
    num_samples = cfg.evaluation.num_samples
    fid = cfg.evaluation.fid

    tg = Swin_TrainerGroup(cfg, rank=0)
    save_path = tg.paths['fid']

    num_iter = num_samples // cfg.training.batch_size + 1
    print(f'Generating samples...')

    # for num in tqdm(range(num_iter)):
        # result = tg.sample(save=False)
        # result = result[-1]
        # result = ((result.clamp(-1, 1) + 1 ) * 127.5).type(torch.uint8)
    for data, _ in tg.dataloader:
        data = data.to(tg.device)
        gaussian, _, _ = tg.downsampler.transform(data)
        result = tg.sample_sr(gaussian[0], shift=False)
        break

        # if fid:
        #     torch.save(result, os.path.join(save_path, f'samples_{num}.pth'))
        # else:
        #     utils.save_images(result, os.path.join(save_path, f'{num}.jpg'))


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