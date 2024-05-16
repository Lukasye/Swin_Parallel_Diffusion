import copy
import os
from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torchvision.utils
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch import optim
from torch.functional import F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
from model.unet import UNet

import matplotlib.pyplot as plt


class Diffusion:
    def __init__(
            self,
            device: str,
            img_size: int,
            noise_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
    ):
        self.device = device
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        # Section 2, equation 4 and near explation for alpha, alpha hat, beta.
        self.beta = self.linear_noise_schedule()
        # self.beta = self.cosine_beta_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Section 3.2, algorithm 1 formula implementation. Generate values early reuse later.
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        # Section 3.2, equation 2 precalculation values.
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)

        # Clean up unnecessary values.
        del self.alpha
        del self.alpha_hat

    def linear_noise_schedule(self) -> torch.Tensor:
        """Same amount of noise is applied each step. Weakness is near end steps image is so noisy it is hard make
        out information. So noise removal is also very small amount, so it takes more steps to generate clear image.
        """
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps, device=self.device)

    def cosine_beta_schedule(self, s=0.008):
        """Cosine schedule from annotated transformers.
        """
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.

        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1).contiguous()
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1).contiguous()
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Random timestep for each sample in a batch. Timesteps selected from [1, noise_steps].
        """
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size, ), device=self.device)

    def p_sample(self, eps_model: nn.Module, n: int, condition: torch.Tensor=None) -> torch.Tensor:
        """Implementation of algorithm 2 sampling. Reverse process, defined by `p` in section 2. Short
         formula is defined in equation 11 of section 3.2.

        From noise generates image step by step. From noise_steps, (noise_steps - 1), ...., 2, 1.
        Here, alpha = 1 - beta. So, beta = 1 - alpha.

        Sample noise from normal distribution of timestep t > 1, else noise is 0. Before returning values
        are clamped to [-1, 1] and converted to pixel values [0, 255].

        Args:
            scale_factor: Scales the output image by the factor.
            eps_model: Noise prediction model. `eps_theta(x_t, t)` in paper. Theta is the model parameters.
            n: Number of samples to process.

        Returns:
            Generated denoised image.
        """

        eps_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

            for i in reversed(range(1, self.noise_steps)):

                t = torch.ones(n, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps_model(x, t, condition)))) +\
                    (epsilon_t * random_noise)

        eps_model.train()


        return x

    def p_sample_single_step(self, x: torch.Tensor, eps_model: nn.Module, timestep: int, condition: torch.Tensor=None) -> torch.Tensor:
        eps_model.eval()
        with torch.no_grad():
            t = torch.ones(x.shape[0], dtype=torch.long, device=self.device) * timestep

            sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
            epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

            random_noise = torch.randn_like(x) if timestep > 1 else torch.zeros_like(x)

            noise_pred, _ = eps_model(x, t, condition)
            result = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * noise_pred))) +\
                (epsilon_t * random_noise)
            
        eps_model.train()
        return result

    def generate_gif(
            self,
            eps_model: nn.Module,
            n: int = 1,
            save_path: str = '',
            output_name: str = None,
            skip_steps: int = 20,
            scale_factor: int = 2,
    ) -> None:
        frames_list = []

        eps_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones(n, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps_model(x, t)))) +\
                    (epsilon_t * random_noise)

                if i % skip_steps == 0:
                    x_img = F.interpolate(input=x, scale_factor=scale_factor, mode='nearest-exact')
                    x_img = ((x_img.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
                    grid = torchvision.utils.make_grid(x_img)
                    img_arr = grid.permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray(img_arr)
                    frames_list.append(img)

        eps_model.train()

        output_name = output_name if output_name else 'output'
        frames_list[0].save(
            os.path.join(save_path, f'{output_name}.gif'),
            save_all=True,
            append_images=frames_list[1:],
            optimize=False,
            duration=80,
            loop=0
        )


class EMA:
    def __init__(self, beta):
        """Modifies exponential moving average model.
        """
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weights, new_weights = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weights=old_weights, new_weights=new_weights)

    def update_average(self, old_weights: torch.Tensor, new_weights: torch.Tensor) -> torch.Tensor:
        if old_weights is None:
            return new_weights
        return old_weights * self.beta + (1 - self.beta) * new_weights

    def ema_step(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 2000) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model=ema_model, model=model)
            self.step += 1
            return
        self.update_model_average(ema_model=ema_model, current_model=model)
        self.step += 1

    @staticmethod
    def reset_parameters(ema_model: nn.Module, model: nn.Module) -> None:
        ema_model.load_state_dict(model.state_dict())


class CustomImageClassDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int,
            image_channels: int
    ):
        super(CustomImageClassDataset, self).__init__()
        self.root_dir = root_dir
        self.class_list = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

        self.image_labels_files_list = list()
        for idx, class_name_folder in enumerate(self.class_list):
            class_path = os.path.join(root_dir, class_name_folder)
            files_list = os.listdir(class_path)
            for image_file in files_list:
                self.image_labels_files_list.append(
                    (
                        os.path.join(class_path, image_file),
                        idx,
                    )
                )

        self.image_files_list_len = len(self.image_labels_files_list)

    def __len__(self) -> int:
        return self.image_files_list_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, class_label = self.image_labels_files_list[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        return self.transform(image), class_label


class Trainer:
    def __init__(
            self,
            cfg,
            rank: int,
            image_size: int,
            num_channels: int,
            device,
            fp16: bool = True,
    ):
        self.world_size = cfg.ngpu
        self.num_channels = num_channels
        self.noise_steps = cfg.training.noise_steps
        self.image_size = image_size
        self.num_epochs = cfg.training.num_epochs
        self.fp16 = fp16
        self.res_scale = cfg.training.res_scale
        self.learning_rate = cfg.training.learning_rate

        self.save_every = cfg.training.save_every
        self.device = device
        self.accumulation_iters = cfg.training.accumulation_iters
        self.accumulation_counter = -1
        self.accumulated_minibatch_loss = 0.0
        self.sample_count = cfg.training.sample_count
        self.warm_up = cfg.training.warm_up

        self.unet_model = UNet(image_size=image_size, num_channels=num_channels,
                               rrdb=cfg.model.use_rrdb,
                               num_rrdb_blocks=cfg.model.num_rrdb_blocks,
                               num_layer=cfg.model.num_layer).to(device)
        if self.device.type != 'cpu':
            self.unet_model = DDP(self.unet_model, device_ids=[rank], static_graph=True)
        self.diffusion = Diffusion(img_size=image_size, 
                                   device=self.device, 
                                   noise_steps=cfg.training.noise_steps,
                                   beta_end=cfg.training.beta_end, 
                                   beta_start=cfg.training.beta_start)
        self.optimizer = optim.Adam(
            params=self.unet_model.parameters(), lr=self.learning_rate,  # betas=(0.9, 0.999)
        )

        # Training scheduler
        num_warmup = cfg.training.warm_up
        warmup = lambda current_step: current_step / num_warmup

        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=warmup)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=200)

        # self.loss_fn = nn.MSELoss().to(self.device)
        self.grad_scaler = GradScaler()

        self.ema = EMA(beta=0.95)
        self.ema_model = copy.deepcopy(self.unet_model).eval().requires_grad_(False)

        self.start_epoch = 0


    def sample(
            self,
            condition: torch.Tensor=None,
            sample_count: int = 1,
    ) -> torch.Tensor:
        """Generates images with reverse process based on sampling method with both training model and ema model.
        """
        # sampled_images = self.diffusion.p_sample(eps_model=self.unet_model, n=sample_count, condition=condition)
        sample = self.diffusion.p_sample(eps_model=self.ema_model, n=sample_count, condition=condition)
        return sample

    def sample_step(self,
                    input_tensor: torch.Tensor,
                    timestep: int,
                    condition: torch.Tensor):
        # return self.diffusion.p_sample_single_step(input_tensor, eps_model=self.ema_model, timestep=timestep, condition=condition)
        return self.diffusion.p_sample_single_step(input_tensor, eps_model=self.ema_model, timestep=timestep, condition=condition)


    def sample_gif(
            self,
            save_path: str = '',
            sample_count: int = 1,
            output_name: str = None,
    ) -> None:
        """Generates images with reverse process based on sampling method with both training model and ema model.
        """
        self.diffusion.generate_gif(
            eps_model=self.unet_model,
            n=sample_count,
            save_path=save_path,
            output_name=output_name,
        )
        self.diffusion.generate_gif(
            eps_model=self.ema_model,
            n=sample_count,
            save_path=save_path,
            output_name=f'{output_name}_ema',
        )

    def train_one_step(self, real_images:torch.Tensor, 
                       ground_truth: torch.Tensor,
                       condition:torch.Tensor=None,
                       time_step: torch.Tensor=None,
                       ext_noise: torch.tensor=None,):
        """ Execute a single training step.

        Args:
            real_images (torch.Tensor): with size (bs, c, w, h)
            condition (torch.Tensor, optional): whether this level should conditioned on the previous one. Defaults to None.

        Returns:
            float: loss
        """
        real_images = real_images.to(self.device)
        if time_step is None:
            current_batch_size = real_images.shape[0]
            time_step = self.diffusion.sample_timesteps(batch_size=current_batch_size)

        # Whether the denoising procedure is finished somewhere else
        if ext_noise is None:
            x_t, noise = self.diffusion.q_sample(x=real_images, t=time_step)
        else:
            x_t = real_images
            noise = ext_noise

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):

            predicted_noise, rrdb_out = self.unet_model(x=x_t, t=time_step, condition=condition)

            loss = F.smooth_l1_loss(predicted_noise, noise) + F.smooth_l1_loss(rrdb_out, rrdb_out)

            self.accumulated_minibatch_loss += loss.detach()

        self.grad_scaler.scale(loss).backward()

        return_loss = None
        # Update the loss at the accumulation step
        if (self.accumulation_counter + 1) % self.accumulation_iters == 0:

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.ema.ema_step(ema_model=self.ema_model, model=self.unet_model)

            return_loss = self.accumulated_minibatch_loss
            self.accumulated_minibatch_loss = 0.0
            self.accumulation_counter = -1

            if self.warmup_scheduler.last_epoch < self.warm_up:
                self.warmup_scheduler.step()

        self.accumulation_counter += 1

        return return_loss 


    def snapshot(self):
        """ Generate a snapshot for the diffusor

        Returns:
            dict: a checkpoint dictionary for this scale
        """
        return utils.save_checkpoint(
            model=self.unet_model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            grad_scaler=self.grad_scaler,
        )
    

    def load_checkpoint(self, checkpoint) -> int:
        self.unet_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'grad_scaler' in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler'])


    def forward_test(self, batch, step_size: int = 50):
        batch = batch.to(self.device)
        for i in range(0, self.noise_steps, step_size):
            t = torch.ones(size=(batch.shape[0], )) * i
            t = t.long().to(self.device)
            x_t, _ = self.diffusion.q_sample(batch, t)

            plt.figure()
            plt.imshow(x_t[0].permute(1, 2, 0).detach().cpu())
            plt.savefig(os.path.join(self.save_path, f'forward_sample_{i}.jpg'))
            plt.close()

    
    def print_model_info(self):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        total_memory = 0

        for param in self.unet_model.parameters():
            # Params multiply the size of the tensor
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params

            # Calculate memory usage assuming the parameter dtype is float32
            # Usually, each float32 takes 4 bytes
            param_memory = num_params * param.element_size()  # element_size() returns the size in bytes of each element
            total_memory += param_memory
        # Convert bytes to megabytes
        total_memory_mb = total_memory / (1024 ** 2)
        return total_params, trainable_params, non_trainable_params, total_memory_mb

    def res2img(self, img_, img_lr_up, clip_input: bool=True):
        if clip_input:
            img_ = img_.clamp(-1, 1)
        img_ = img_ / self.res_scale + img_lr_up
        return img_

    def img2res(self, x, img_lr_up, clip_input:bool=True):
        x = (x - img_lr_up) * self.res_scale
        if clip_input:
            x = x.clamp(-1, 1)
        return x