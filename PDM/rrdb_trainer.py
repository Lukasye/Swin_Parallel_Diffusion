import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random

import torch
import torch.nn.functional as F
from torch import optim
import hydra
from hydra.core.hydra_config import HydraConfig

import utils
from model.rrdb import RRDBNet
from data_utils.SRDataloader import get_celeba_dataloader


class RRDB_Trainer:

    def __init__(self, cfg, 
                 device,
                 fp16: bool = True) -> None:
        self.device = device
        self.fp16 = fp16

        self.image_size = 128
        self.num_channels = 3
        self.scaling_factor = cfg.training.scaling_factor
        self.diffuser_kernel_size = self.image_size // self.scaling_factor
        self.crop_size = self.image_size
        self.batch_size = cfg.training.batch_size
        self.eval_batch_size = cfg.evaluation.batch_size

        self.shift = cfg.group.shift
        self.rolling_pattern = cfg.group.rolling_pattern
        self.repeats = self.scaling_factor ** 2
        self.pattern = [(0, 0), (0, self.shift), (self.shift, self.shift), (self.shift, 0)]
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.scaling_factor, stride=self.scaling_factor)
        self.attention_mask = self.generate_attention_mask().to(self.device)

        self.rrdb = RRDBNet(input_channel=cfg.rrdb.cond_channel,
                                        num_feature=cfg.rrdb.num_feature, 
                                        output_channel=3,
                                        num_of_blocks=cfg.model.num_rrdb_blocks, 
                                        scaling_factor=cfg.training.scaling_factor).to(device)
        self.optimizer = optim.Adam(
            params=self.rrdb.parameters(), lr=cfg.rrdb.learning_rate,  # betas=(0.9, 0.999)
        )

        # Training scheduler
        self.warm_up = cfg.training.warm_up
        warmup = lambda current_step: current_step / self.warm_up

        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=warmup)

        self.dataloader, self.sample_dataloader = get_celeba_dataloader('/bigpool/homes/yeyun/projects/PyramidDiffusionModel/datasets/celeba/resize_128',
                                                                        batch_size=self.batch_size, val_batch_size=self.eval_batch_size, 
                                                                        scale=self.scaling_factor, augmentation=False)

    def save_rrdb(self, save_path):
        checkpoint = {'state_dict': self.rrdb.state_dict()}
        torch.save(checkpoint, save_path)

    def train_rrdb(self):
        print('start training rrdb block...')

        total_iter_per_batch = len(self.dataloader)
        sample_iter = iter(self.sample_dataloader)
        counter = 0

        while True:
            train_iter = iter(self.dataloader)
            counter = 0

            while True:
                if counter >= total_iter_per_batch:
                    break
                batch_idx = (torch.randint(0, self.repeats, size=(self.batch_size, )) + torch.arange(self.batch_size) * self.repeats).type(torch.long)
                batch = next(train_iter)
                batch = self.all_to_device(batch)
                gt, lr, _ = batch
                roll_pattern = self.generate_rolling_pattern()
                inverse_roll_pattern = tuple(-shift for shift in roll_pattern)

                input_tensor = self.generate_rolling_condition(lr, attention_shift=inverse_roll_pattern)
                target = torch.roll(gt, shifts=roll_pattern, dims=(-2, -1))
                target = self.patchify(target)
                input_tensor = input_tensor[batch_idx, ::]
                target = target[batch_idx, ::]
                
                loss, lr = self.train_one_step(input_tensor, target)

                if counter % 40 == 0:
                    print(f'Iteration: {counter}\t Loss: {loss} \t lr: {lr}')

                if counter % 500 == 0:
                    _, lr, _ = self.all_to_device(next(sample_iter))
                    roll_pattern = self.generate_rolling_pattern()
                    inverse_roll_pattern = tuple(-shift for shift in roll_pattern)

                    input_tensor = self.generate_rolling_condition(lr, attention_shift=inverse_roll_pattern, train=False)
                    sample = self.rrdb_trainer.sample(input_tensor)
                    utils.save_images(sample, f'./sample_{counter}.jpg', unnormalized=True)
                    self.rrdb_trainer.save_rrdb(os.path.join('./', f"rrdb_{counter}.pth"))
                
                counter += 1

    def train_one_step(self, input_tensor, target):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            pred, _ = self.rrdb(input_tensor)
        
            loss = F.smooth_l1_loss(pred, target, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.warmup_scheduler.last_epoch < self.warm_up:
            self.warmup_scheduler.step()

        return loss.item(), self.warmup_scheduler.get_last_lr()[0]
        
    @ torch.no_grad()
    def sample(self, input_tensor):
        out, _ = self.rrdb(input_tensor)
        return out

    def all_to_device(self, batch: tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = x.to(self.device)
        return tuple(batch)

    def generate_rolling_pattern(self, rolling_pattern = 'fix_pattern') -> tuple:
        if rolling_pattern == 'fix_pattern':
            roll_index = torch.randint(0, 4, size=(1, ))
            return self.pattern[roll_index]
        elif rolling_pattern == 'random':
            return random.randint(0, self.shift), random.randint(0, self.shift)
        elif rolling_pattern == 'no':
            return 0, 0
        else:
            raise NotImplementedError('This kind of rolling pattern is not implemented!')

    @ torch.no_grad()
    def generate_rolling_condition(self, glob, attention_shift: tuple, train:bool=True):
        # attention_mask = self.attention_mask if train else  self.attention_mask[:self.eval_batch_size * self.scaling_factor ** 2,:,:,:]
        # Prepare the attention mask according to the batch size
        num_repeat = self.batch_size if train else self.eval_batch_size
        attention_mask = torch.cat(num_repeat * [self.attention_mask])
        glob_cond = torch.repeat_interleave(glob, repeats=self.repeats, dim=0)

        rolled_attention = torch.roll(attention_mask, shifts=attention_shift, dims=(-2, -1))
        rolled_attention = self.avg_pool(rolled_attention)  # down scale the attention mask with scaling factor
        return torch.cat((glob_cond, rolled_attention), dim=1)

    def generate_attention_mask(self) -> torch.Tensor:
        """ generate attention mask for a single batches, since the attention mask won't change,
            it will be part of the initialization .

        Args:
            normalized (bool, optional): whether the result will be normalized. Defaults to True.

        Returns:
            torch.Tensor: attention mask with shape (self.batch_size * self.scaling_factor ** 2, 1, self.crop_size, self.crop_size)
        """
        attention_mask = torch.empty((self.repeats, self.crop_size, self.crop_size))
        half_kernel_size = self.diffuser_kernel_size // 2
        xx, yy = torch.meshgrid(torch.arange(0, (self.crop_size + half_kernel_size) * 2), torch.arange(0, (self.crop_size + half_kernel_size) * 2), indexing='ij')
        kernel = torch.exp(-((xx - self.crop_size)**2 + (yy - self.crop_size)**2) / (2 * (half_kernel_size * 1.5)**2))
        
        for i in range(self.scaling_factor):
            for j in range(self.scaling_factor):
                tmp = kernel.clone()
                top = self.crop_size - i * self.diffuser_kernel_size - half_kernel_size
                left = self.crop_size - j * self.diffuser_kernel_size - half_kernel_size
                tmp = tmp[top: top + self.crop_size, left: left + self.crop_size]

                tmp = tmp / torch.max(tmp) * 2. - 1.

                attention_mask[i * self.scaling_factor + j] = tmp
        
        # attention_mask = torch.cat(self.batch_size * [attention_mask]).unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def patchify(self, imgs:torch.Tensor):

        kernel_size = self.diffuser_kernel_size
        bs, c, image_size, _ = imgs.shape  # we assume same width and height

        assert image_size % kernel_size == 0  # the image must be perfektly dividable 

        h_k = image_size // kernel_size

        imgs = imgs.view(bs, c, h_k, kernel_size, h_k, kernel_size)
        imgs = imgs.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = imgs.view(bs * h_k * h_k, c, kernel_size, kernel_size)

        return patches

    def unpatchify(self, patches, bs=None):
        if bs is None:
            assert patches.shape[0] % self.scaling_factor ** 2 == 0
            bs = patches.shape[0] // self.scaling_factor ** 2

        kernel_size = self.diffuser_kernel_size
        h_k = self.crop_size // kernel_size

        # Reshape patches to the original grid layout
        imgs = patches.view(bs, h_k, h_k, self.num_channels, kernel_size, kernel_size)
        imgs = imgs.permute(0, 3, 1, 4, 2, 5).contiguous()
        imgs = imgs.view(bs, self.num_channels, self.crop_size, self.crop_size)

        return imgs

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir 
    cfg.work_dir = work_dir

    device = torch.device('cpu')
    trainer = RRDB_Trainer(cfg, device)
    trainer.train_rrdb()

if __name__ == "__main__":
    main()