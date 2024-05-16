import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import matplotlib.pyplot as plt
import numpy as np

from ddpm_group import get_trainergroup, Swin_TrainerGroup, TrainerGroup
from utils import get_device
from model.rrdb import RRDBNet
from data_utils import celeba_dataloader
import utils



def rolling_test(tg: Swin_TrainerGroup, num_samples: int = 5):
    data, _, _ = next(iter(tg.dataloader))
    pattern_idx = 2
    roll_pattern = tg.pattern[pattern_idx]
    print(f'Roll pattern: {roll_pattern}')
    rolled_batch = torch.roll(data, shifts=roll_pattern, dims=(-2, -1))
    utils.save_images(images=rolled_batch, unnormalized=True, save_path=os.path.join(tg.paths['test'], f'rolled.jpg'))
    inverse_roll_pattern = tuple(-shift for shift in roll_pattern)
    unrolled_batch = torch.roll(rolled_batch, shifts=inverse_roll_pattern, dims=(-2, -1))
    utils.save_images(images=unrolled_batch, unnormalized=True, save_path=os.path.join(tg.paths['test'], f'unrolled.jpg'))


def pyramid_test(tg: Swin_TrainerGroup):
    print('Running swin downsample test...')
    data, _, _ = next(iter(tg.dataloader))
    gaussian, laplace, _ = tg.downsampler.transform(data)
    utils.save_images(images=gaussian[0], unnormalized=True, save_path=os.path.join(tg.paths['test'], f'condition.jpg'))
    utils.save_images(images=laplace[0], unnormalized=True, save_path=os.path.join(tg.paths['test'], f'ground_truth.jpg'))


def swin_forward_test(tg: Swin_TrainerGroup, step_size: int = 10,
                      num_sampless: int = 2):
    print('Running swin forward diffusion test...')
    data, _, lr_up = next(iter(tg.dataloader))
    data = data.to(tg.device)
    lr_up = lr_up.to(tg.device)
    res_org = tg.swin_diffuser.img2res(data, lr_up)
    for j in range(num_sampless):
        results = []
        results_real = []
        # gaus, lapl, _ = tg.downsampler.transform(data[j].unsqueeze(0))
        # foo = lapl[0]
        foo = res_org[j, ::].unsqueeze(0)
        # foo = tg.crop_image(foo, 0)
        # bg = tg.crop_image(gaus[0], 0)
        foos = tg.patchify(foo)
        # foo = gaus[-1] + lapl[-1]
        for i in range(0, tg.noise_steps , step_size):
            t = torch.ones(size=(foos.shape[0], )) * i
            t = t.long()
            # x_t, _ = tg.trainers[0].diffusion.q_sample(foo, t)
            x_t, _ = tg.swin_diffuser.diffusion.q_sample(foos, t)

            res = tg.unpatchify(x_t)
            results.append(res)
            results_real.append(tg.swin_diffuser.res2img(res, lr_up[j, ::]))
        utils.save_images(images=torch.concat(results), unnormalized=True, save_path=os.path.join(tg.paths['test'], f'forward_sample_res{j}.jpg'))
        utils.save_images(images=torch.concat(results_real), unnormalized=True, save_path=os.path.join(tg.paths['test'], f'forward_sample{j}.jpg'))

def pachify_test(tg: Swin_TrainerGroup):
    print('Running swin patchify and unpatchfy test...')
    num_diffuser = tg.scaling_factor
    data_uncroped, _, _ = next(iter(tg.dataloader))
    data = tg.crop_image(data_uncroped, pattern_index=0)
    print('After crop: ', data.shape)
    results = tg.patchify(data)
    print('Shifting: ', tg.shift)
    print('Final shape: ', results.shape)
    fig, axes = plt.subplots(num_diffuser, num_diffuser, figsize=(15, 15))
    for i in range(num_diffuser):
        for j in range(num_diffuser):
            ax = axes[i, j]
            result = results[i * num_diffuser + j]
            result = (result + 1) * 127.5
            result = result.type(torch.uint8)
            ax.imshow(result.permute(1, 2, 0).numpy())
            ax.set_title(f'Image ({i+1}, {j+1})')
            ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(tg.paths['test'], 'patchify.jpg'))
    plt.close()

    plt.figure()
    unpachify = tg.unpatchify(results, bs=tg.batch_size)
    asd = unpachify[0].permute(1, 2, 0)
    asd = (asd + 1) * 127.5
    asd = asd.type(torch.uint8)
    plt.imshow(asd)
    plt.savefig(os.path.join(tg.paths['test'], 'unpatchify.jpg'))
    plt.close()
    

    plt.figure()
    results /= 0.5
    refill = tg.refill_image(data_uncroped, results, pattern_index=0)
    qwe = refill[0].permute(1, 2, 0)
    qwe = (qwe + 1) * 127.5
    qwe = qwe.type(torch.uint8)
    plt.imshow(qwe)
    plt.savefig(os.path.join(tg.paths['test'], 'refill.jpg'))
    plt.close()
        

def gaussian_kernel_test(tg: Swin_TrainerGroup):
    print('Generating attention mask sample...')
    num = tg.scaling_factor
    results = tg.generate_attention_mask()
    bached_result = results
    # bached_result = torch.cat(3*[results])
    print(results.shape)
    fig, axes = plt.subplots(num * 2, num , figsize=(15, 30))
    for i in range(num * 2):
        for j in range(num):
            ax = axes[i, j]
            result = bached_result[i * num + j]
            ax.imshow(result.permute(1, 2, 0).numpy())
            ax.set_title(f'Image ({i+1}, {j+1})')
            ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(tg.paths['test'], 'attention_mask_demo.jpg'))
    plt.close()

def crop_shift_test(tg: Swin_TrainerGroup):
    data, _, _ = next(iter(tg.dataloader))
    results = []
    fig, axes = plt.subplots(2, 2 , figsize=(15, 30))
    for i in range(2):
        for j in range(2):
            results.append(tg.crop_image(data, pattern_index=i * 2 + j))
            ax = axes[i, j]
            result = results[i * 2 + j][0]
            print(result.shape)
            assert result.shape == (3, tg.crop_size, tg.crop_size)
            result = (result + 1) * 127.5
            result = result.type(torch.uint8)
            ax.imshow(result.permute(1, 2, 0).numpy())
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('crop_shift_demo.jpg')
    plt.close()
    
    
def rrbn_test(tg):
    data, _, _ = next(iter(tg.dataloader))
    print(f'Input dim: {data.shape}')
    model = RRDBNet(input_channel=3, num_feature=32, output_channel=3, num_of_blocks=3)
    out, cond = model(data)
    print(f'Output dim:{out.shape}')
    print(f'cond dim:{cond.shape}')
    

def swin_test(tg: Swin_TrainerGroup):
    batch, _, _ = next(iter(tg.dataloader))
    tg.sample_sr(batch, save_path='./')



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir 
    cfg.work_dir = work_dir
    cfg.ngpu = 1

    tg = get_trainergroup(cfg, -1, init_test=False)
    print(f'Epoch: {tg.epoch}')
    print(f'Iteration: {tg.iteration_counter}')
    print(f'Length of the dataloader: {len(tg.dataloader)}')
    # swin_forward_test(tg)
    # pyramid_test(tg)
    # pachify_test(tg)
    # crop_shift_test(tg)
    # gaussian_kernel_test(tg)
    rrbn_test(tg)
    # swin_test(tg)
    # rolling_test(tg)


if __name__ == '__main__':
    main()