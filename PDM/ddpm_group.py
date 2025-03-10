import pickle
import os
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics.image.fid import FrechetInceptionDistance
from torch import distributed as dist
import torch.nn.functional as F
from abc import abstractmethod, ABC
from pytorch_wavelets import DWTForward, DWTInverse
# from ignite.metrics import FID

from data_utils import SRDataloader
from ddpm import Trainer
from downsampler import LaplacianDownsampler
import utils
from metrics.sr_metrics import Measure



class TrainerGroup(ABC):
    def __init__(self,
                 cfg,
                 rank,
                 init_test,
                 customer_dataloader = None,
                 train: bool = True,
                 memroy_check: bool = True
                 ) -> None:
        self.init_test = init_test
        self.rank = rank
        self.world_size = cfg.ngpu
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu") if rank > -1 else torch.device("cpu")
        self.train_mod = train
        self.memory_check = memroy_check and self.rank == 0

        self.batch_size = cfg.training.batch_size // self.world_size
        self.eval_batch_size = cfg.evaluation.batch_size
        self.num_epochs = cfg.training.num_epochs
        self.dataset = cfg.training.dataset
        self.save_every = cfg.training.save_every
        self.record_loss_every = cfg.training.record_loss_every
        self.iteration_counter =  0
        self.img_downsample_factor = cfg.training.scaling_factor
        self.noise_steps = cfg.training.noise_steps
        self.epoch = 0
        self.current_loss = 0.

        self.paths, self.logger = self.prepare_working_env(cfg.work_dir)
        (self.dataloader, self.sample_dataloader), self.image_size, self.num_channels = self.prepare_dataset(cfg, batch_size=self.batch_size,
                                                                                   val_batch_size=self.eval_batch_size, external_dataloader=customer_dataloader)

    def prepare_working_env(self, work_dir: str):
        paths = {}
        for dir in ['samples', 'checkpoints', 'checkpoint_meta', 'fid', 'test']:
            paths[dir] = os.path.join(work_dir, dir)
            if not os.path.exists(paths[dir]) and self.rank <= 0:
                os.mkdir(paths[dir])
        logger = utils.get_logger(os.path.join(work_dir, 'logs')) if self.rank == 0 else None
        return paths, logger

    def prepare_dataset(self, cfg: str, 
                        batch_size: int,
                        val_batch_size: int,
                        external_dataloader: DataLoader = None) -> DataLoader:
        assert cfg.training.batch_size % cfg.ngpu == 0
        # generator = torch.Generator().manual_seed(2024)
        dataset_name = cfg.training.dataset
        if external_dataloader is not None and  dataset_name == 'CUSTOM':
            return external_dataloader, next(iter(external_dataloader)).shape[-1], 3
        elif dataset_name == 'CIFAR':
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)])
            diffusion_dataset = torchvision.datasets.CIFAR10(cfg.data.dataset_path,
                                                             transform=data_transforms,
                                                             train=True,
                                                             download=True)
            train, val = random_split(diffusion_dataset, lengths=[0.9, 0.1])
            train_loader = DataLoader(train,batch_size=batch_size,shuffle=True,
                                      pin_memory=True,num_workers=cfg.training.num_workers,drop_last=False,collate_fn=utils.collate_fn)
            val_loader = DataLoader(val,batch_size=batch_size,shuffle=True,
                                      pin_memory=True,num_workers=cfg.training.num_workers,drop_last=False,collate_fn=utils.collate_fn)
            return train_loader, val_loader, 32, 3
        elif dataset_name == 'IMAGENET':
            size = 64
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((size, size)),
                transforms.Lambda(lambda t: (t * 2) - 1)])
            diffusion_dataset = torchvision.datasets.ImageNet(cfg.data.dataset_path,
                                                              transform=data_transforms)
            train, val = random_split(diffusion_dataset, lengths=[0.9, 0.1])
            train_loader = DataLoader(train,batch_size=batch_size,shuffle=True,
                                      pin_memory=True,num_workers=cfg.training.num_workers,drop_last=False,collate_fn=utils.collate_fn)
            val_loader = DataLoader(val,batch_size=batch_size,shuffle=True,
                                      pin_memory=True,num_workers=cfg.training.num_workers,drop_last=False,collate_fn=utils.collate_fn)
            return (train_loader, val_loader), size, 3
        elif dataset_name.startswith('CELEBA'):
            size = int(dataset_name.split('_')[1])
            return SRDataloader.get_celeba_dataloader(os.path.join(cfg.data.dataset_path, 'celeba', f'resize_{size}'),
                                         batch_size=batch_size, val_batch_size=val_batch_size, 
                                         scale=self.img_downsample_factor, 
                                         augmentation=cfg.data.augmentation), size, 3
        else:
            raise NotImplementedError(f'No such kind of Dataset {dataset_name}')

################################################# GENERAL FUNCTION #########################################
    @ abstractmethod
    def save_checkpoints(self):
        """ Save all the diffusors

        Returns:
            None
        """
        pass

    @ abstractmethod
    def load_checkpoints(self, filename: str):
        """ load all the diffusor

        Args:
            filename (str): path to the pth file
        """
        pass

    @ abstractmethod
    def train(self):
        """ implemente training procedure
        """
        pass

    @ abstractmethod
    def train_one_step(self, batch: tuple) -> float:
        """ one training step

        Args:
            batch (torch.Tensor): input images
        """
        pass

    @ abstractmethod
    def sample(self, save: bool=True):
        """sample a batch of images with the current NN

        Args:
            save (bool, optional): Whether the outcome is saved asa jpg images.

        Returns:
            generate sample with shape (bs, c, w, h)
        """
        pass

    @abstractmethod
    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        pass

    @abstractmethod
    def print_basic_info(self):
        pass

################################################# HELPER FUNCTION #########################################

    def log_print(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)

    def print_model_info(self):

        self.log_print("***************************** Model Info *********************************")
        for trainer in self.trainers:
            total_params, trainable_params, non_trainable_params, total_memory_mb = trainer.print_model_info()
            self.log_print(f"Total params: {total_params}")
            self.log_print(f"Trainable params: {trainable_params}")
            self.log_print(f"Non-trainable params: {non_trainable_params}")
            self.log_print(f"Estimated Total Memory Usage: {total_memory_mb:.2f} MB")

    def print_device_info(self):
        self.log_print("***************************** Device Info *********************************")
        if self.device.type == "cuda":
            num_devices = torch.cuda.device_count()
            for i in range(num_devices):
                device = torch.cuda.get_device_properties(i)
                self.log_print(f"Device {i}: {device.name}")
                self.log_print(f"  Memory: {device.total_memory / 1e9} GB")
                self.log_print(f"  Compute Capability: {device.major}.{device.minor}")
                self.log_print(f"  Multi-processors: {device.multi_processor_count}")
        else:
            self.log_print("WARNING: Using device {}".format(self.device))

    def all_to_device(self, batch: tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = x.to(self.device)
        return tuple(batch)

class Pyramid_TrainerGroup(TrainerGroup):
    """
    Class to coordinate different diffusor.
    """
    def __init__(self, cfg, 
                 rank, 
                 init_test = True,
                 train: bool = True,
                 ) -> None:
        super().__init__(cfg, rank, init_test, train=train)
        self.num_diffuser = cfg.group.num_diffuser
        self.trainers = [Trainer(cfg, int(self.image_size/(i+1)), num_channels=self.num_channels, device=self.device) 
                         for i in range(self.num_diffuser)]
        self.downsampler = LaplacianDownsampler(self.num_diffuser, device=self.device)
        self.fid = FrechetInceptionDistance(feature=2048)

        # load checkpoint
        if len(os.listdir(self.paths['checkpoint_meta'])) != 0:
            self.load_checkpoints(os.path.join(self.paths['checkpoint_meta'], "checkpoint_meta.pth"))
        else:
            self.log_print('No checkpoint found, starting from scratch.')
        


    def save_checkpoints(self):
        checkpoints = {}
        for num, trainer in enumerate(self.trainers):
            checkpoints[num] = trainer.snapshot()
        checkpoints['Iteration'] = self.iteration_counter
        torch.save(checkpoints, os.path.join(self.paths['checkpoints'],
                                             f"{self.dataset}_{self.iteration_counter}.pth"))
        torch.save(checkpoints, os.path.join(self.paths['checkpoint_meta'],
                                             f"checkpoint_meta.pth"))


    def load_checkpoints(self, filename: str):
        checkpoints = torch.load(filename, map_location="cuda")
        self.iteration_counter = checkpoints['Iteration']
        for num in range(self.num_diffuser):
            self.trainers[num].load_checkpoint(checkpoints[num])
        self.log_print(f"Checkpoint loaded from {filename}!")
        self.log_print(f"Current iterarion: {self.iteration_counter}")


################################################# TRAINING FUNCTION #########################################

    def train(self):
        self.log_print('Starting training procedure...')
        self.print_device_info()
        self.print_model_info()
        self.log_print("***************************** Iteration *********************************")

        while self.iteration_counter < self.num_epochs:

            for _, (batch, _) in enumerate(self.dataloader):
                loss, fids = self.train_one_step(batch)

                if self.iteration_counter % self.record_loss_every == 0:
                    self.log_print(f'Iteration: {self.iteration_counter}.\tLoss: {loss}')

                if self.iteration_counter % self.save_every == 0:
                    self.save_checkpoints()
                    self.sample()
                    # fid_score = self.evaluate(fids)
                    # self.log_print(f"Evaluation results: FID: {fid_score}")

                    dist.barrier()

                self.iteration_counter += 1

            self.scheduler_update()


    def train_one_step(self, batch: torch.Tensor):
        """ Execute one single training step, might not update because the accumulation

        Args:
            batch (torch.Tensor): with size (bs, c, w, h)

        Returns:
            loss_record (list): loss of different scale diffusor
        """
        counter = self.num_diffuser
        loss_record = []
        gaussians, laplacians, fids = self.downsampler.transform(batch)

        for trainer, gaussian, laplacian in zip(self.trainers, gaussians, laplacians):
            # If it's already the last layer then do normal diffusion
            loss = trainer.train_one_step(gaussian + laplacian) if counter == 1 else trainer.train_one_step(laplacian, gaussian)
            loss_record.append(loss)
            counter -= 1

        return loss_record, fids


    def scheduler_update(self):
        for trainer in self.trainers:
            trainer.scheduler.step()

################################################# SAMPLE FUNCTION #########################################

    def sample(self, save: bool=True) -> list:
        """sample a batch of images with the current NN

        Args:
            save (bool, optional): Whether the outcome is saved asa jpg images.

        Returns:
            list : list of samples generated in different scale
        """

        self.log_print(f"Generating sample at epoch {self.epoch} iteration step: {self.iteration_counter}")
        sample = self.trainers[-1].sample(sample_count=self.eval_batch_size)
        result = [sample.clone()]

        if save:
            x = ((sample.clamp(-1, 1) + 1 ) * 127.5).type(torch.uint8)
            utils.save_images(x, os.path.join(self.paths['samples'], f'first_layer_{self.iteration_counter}.jpg'))

        for trainer in self.trainers[:-1][::-1]:
            cond = self.downsampler.upsample(sample)
            d_sample = trainer.sample(cond, sample_count=self.eval_batch_size)
            sample = cond + d_sample
            result.append(sample.clone())

        if save:
            x = ((sample.clamp(-1, 1) + 1 ) * 127.5).type(torch.uint8)
            utils.save_images(x, os.path.join(self.paths['samples'], f'{self.iteration_counter}.jpg'))

        return result

################################################# EVALUATION FUNCTION #########################################

    def evaluate(self, pred, target=None):
        fid = []

        with torch.no_grad():
            results = self.sample()

        for result, gaussian in zip(results[::-1], pred):
            fid.append(self.get_fid_score(result, gaussian))
        
        return fid


    def get_fid_score(self, prediction, real_batch, resize = True):
        self.fid.reset()
        if resize:
            real_batch = utils.resize_tensor(real_batch).cpu()
            prediction_resize = utils.resize_tensor(prediction).cpu()
        self.fid.update(real_batch, real=True)
        self.fid.update(prediction_resize, real=False)
        return self.fid.compute()



class Swin_TrainerGroup(TrainerGroup):
    def __init__(self, cfg, rank, 
                 init_test = True,
                 train: bool = True,
                 rrdb_mode: bool = False
                 ) -> None:
        super().__init__(cfg, rank, init_test, train=train)

        self.diffuser_kernel_size = cfg.group.kernel_size
        self.scaling_factor = self.image_size // self.diffuser_kernel_size
        assert self.image_size % self.scaling_factor == 0

        # self.crop_size = self.diffuser_kernel_size * self.scaling_factor
        # self.diffuser_kernel_size = self.image_size // self.scaling_factor
        self.crop_size = self.image_size

        self.shift = cfg.group.shift
        self.rolling_pattern = cfg.group.rolling_pattern
        self.repeats = self.scaling_factor ** 2
        self.pattern = [(0, 0), (0, self.shift), (self.shift, self.shift), (self.shift, 0)]
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.scaling_factor, stride=self.scaling_factor)

        self.swin_diffuser = Trainer(cfg, rank, self.diffuser_kernel_size, num_channels=self.num_channels, device=self.device, train=self.train_mod)
        self.attention_mask = self.generate_attention_mask().to(self.device)
        self.metric = Measure(self.img_downsample_factor)

        # load checkpoint if there's one and not in rrdb training mode
        if len(os.listdir(self.paths['checkpoint_meta'])) != 0 and not rrdb_mode:
            if os.path.exists(os.path.join(self.paths['checkpoint_meta'], "checkpoint_meta.pth")):
                self.load_checkpoints(os.path.join(self.paths['checkpoint_meta'], "checkpoint_meta.pth"))
            # if os.path.exists(os.path.join(self.paths['checkpoint_meta'], "rrdb.pth")):
            #     self.swin_diffuser.load_rrdb(os.path.join(self.paths['checkpoint_meta'], "rrdb.pth"))
        else:
            self.log_print('No checkpoint found, starting from scratch.')

        if self.init_test:
            import structure_test
            structure_test.swin_forward_test(self)
            # structure_test.pachify_test(self)
            # structure_test.gaussian_kernel_test(self)
            structure_test.rolling_test(self)

        
    def print_basic_info(self):
        self.print_device_info()

    def save_checkpoints(self, save_rrdb: bool=False):
        checkpoints = {}
        checkpoints['swin'] = self.swin_diffuser.snapshot()
        checkpoints['Iteration'] = self.iteration_counter
        checkpoints['epoch'] = self.epoch
        torch.save(checkpoints, os.path.join(self.paths['checkpoints'],
                                             f"{self.dataset}_{self.epoch}_{self.iteration_counter}.pth"))
        torch.save(checkpoints, os.path.join(self.paths['checkpoint_meta'],
                                             f"checkpoint_meta.pth"))


    def load_checkpoints(self, filename: str):
        # checkpoints = torch.load(filename, map_location="cuda")
        checkpoints = torch.load(filename, map_location=self.swin_diffuser.device)
        self.iteration_counter = checkpoints['Iteration']
        self.epoch = checkpoints['epoch']
        # self.ordinary_diffuser.load_checkpoint(checkpoints['ordinary'])
        self.swin_diffuser.load_checkpoint(checkpoints['swin'])
        self.log_print(f"Checkpoint loaded from {filename}!")
        self.log_print(f"Current epoch: {self.epoch}")
        self.log_print(f"Current iterarion: {self.iteration_counter}")

    def crop_image(self, batch: torch.Tensor, pattern_index: int) -> torch.Tensor:
        """ crop the input batch to the size that's dividable for kernel size

        Args:
            batch (torch.Tensor): input images with shape (bs, c, image_size, image_size)
            pattern_index (int): which shifting pattern we use. 

        Returns:
            torch.Tensor: images batch with shape (bs, c, self.kernel_size, self.kernel_size)
        """
        assert pattern_index < len(self.pattern)

        top, left = self.pattern[pattern_index]
        return batch[:, :, top:top+self.crop_size, left:left+self.crop_size]

    def refill_image(self, original: torch.Tensor, extra_content: torch.Tensor, pattern_index: int, train: bool = True) -> torch.Tensor:
        """ Refill the original image

        Args:
            original (torch.Tensor): original image with shape (bs, c, org_w, org_h)
            extra_content (torch.Tensor): part to fillin with shape (bs, c, w, h), where w and h must smaller than org_w and oeg_h
            pattern_index (int): the shifting pattern used to crop the image
            train (bool, optional): The batchsize used in train and evaluation is different. Defaults to True.

        Returns:
            torch.Tensor: the original image
        """
        assert pattern_index < len(self.pattern)

        batch_size = self.batch_size if train else self.eval_batch_size

        top, left = self.pattern[pattern_index]
        filling = self.unpatchify(extra_content, batch_size)
        original[:, :, top:top+self.crop_size, left:left+self.crop_size] = filling
        return original

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

    def generate_attention_mask(self, normalized: bool=True) -> torch.Tensor:
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

                # kernel_sum = torch.sum(tmp)
                # if kernel_sum > 0 and normalized:
                #     tmp /= kernel_sum
                tmp = tmp / torch.max(tmp) * 2. - 1.

                attention_mask[i * self.scaling_factor + j] = tmp
        
        # attention_mask = torch.cat(self.batch_size * [attention_mask]).unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def print_basic_info(self):
        self.print_device_info()


    def train(self):
        assert self.swin_diffuser is not None

        self.log_print('Starting training procedure...')
        self.print_device_info()
        self.log_print(f'Optimizer: {self.swin_diffuser.optimizer.state_dict}')
        self.log_print("***************************** Iteration *********************************")
        total_iter_per_batch = len(self.dataloader)
        self.log_print(f'Length of dataloader: {total_iter_per_batch}')
        sample_iter = iter(self.sample_dataloader)
        # mem_counter = 50

        while True:
            train_iter = iter(self.dataloader)

            while True:
                if self.iteration_counter >= total_iter_per_batch:
                    break
                batch = next(train_iter)
                batch = self.all_to_device(batch)

                loss = self.train_one_step(batch)
                self.current_loss = loss if loss is not None else self.current_loss

                if self.iteration_counter % self.record_loss_every == 0:
                    lr = self.swin_diffuser.optimizer.param_groups[-1]['lr']
                    self.log_print(f'Epoch {self.epoch} \tIteration: {self.iteration_counter}\tlr: {lr}\nLoss: {self.current_loss}')

                if (self.iteration_counter) % self.save_every == 0:
                    if self.rank == 0:
                        self.save_checkpoints()
                    batch = next(sample_iter)
                    batch = self.all_to_device(batch)
                    pred = self.sample(batch)
                    self.evaluate(pred, batch)
                
                    if self.world_size > 1:
                        dist.barrier()

                self.iteration_counter += 1

            self.scheduler_update()
            self.log_print(f'Episode {self.epoch} end.')
            self.epoch += 1
            self.iteration_counter = 0

# ********************************************************** Other Method  *****************************************

    def generate_rolling_pattern(self, rolling_pattern = 'fix_pattern') -> tuple:
        if rolling_pattern == 'fix_pattern':
            roll_index = torch.randint(0, 4, size=(1, ))
            return self.pattern[roll_index]
        elif rolling_pattern == 'random':
            return random.randint(0, self.shift), random.randint(0, self.shift)
        elif rolling_pattern == 'still':
            return 0, 0
        else:
            raise NotImplementedError(f'This kind of rolling pattern is not implemented! patterm:{rolling_pattern}')

    def train_one_step(self, batch: tuple):
        """ Train a single step with rolling shift, for super vision type problem

        Args:
            batch (tuple): _description_

        Returns:
            _type_: _description_
        """
        gt, lr, lr_up = batch
        x_0 = self.swin_diffuser.img2res(gt, lr_up)

        # q_sample on the original image and then patchify it
        t = torch.randint(low=1, high=self.noise_steps, size=(self.batch_size, ), device=self.device)
        x_t_noise, noise = self.swin_diffuser.diffusion.q_sample(x=x_0, t=t)
        t = t.repeat_interleave(dim=0, repeats=self.repeats)
        
        # rolling configuration
        roll_pattern = self.generate_rolling_pattern(self.rolling_pattern)
        inverse_roll_pattern = tuple(-shift for shift in roll_pattern)

        rolled_batch = torch.roll(x_t_noise, shifts=roll_pattern, dims=(-2, -1))
        noise = torch.roll(noise, shifts=roll_pattern, dims=(-2, -1))

        input_batch = self.patchify(rolled_batch)
        noise = self.patchify(noise)

        condition = self.generate_rolling_condition(lr, attention_shift=inverse_roll_pattern)

        loss = self.swin_diffuser.train_one_step(input_batch, 
                                                 condition=condition, 
                                                 time_step=t, ext_noise=noise)

        if loss is not None and self.world_size > 1:
            dist.all_reduce(loss)
            loss /= self.world_size

        return loss

    @ torch.no_grad()
    def generate_rolling_condition(self, glob, attention_shift: tuple, train:bool=True):
        # attention_mask = self.attention_mask if train else  self.attention_mask[:self.eval_batch_size * self.scaling_factor ** 2,:,:,:]
        # Prepare the attention mask according to the batch size
        num_repeat = self.batch_size if train else self.eval_batch_size
        attention_mask = torch.cat(num_repeat * [self.attention_mask])
        rolled_attention = torch.roll(attention_mask, shifts=attention_shift, dims=(-2, -1))
        rolled_attention = self.avg_pool(rolled_attention)  # down scale the attention mask with scaling factor

        glob_cond = torch.repeat_interleave(glob, repeats=self.repeats, dim=0)
        if self.img_downsample_factor != self.scaling_factor:
            glob_cond = torch.nn.functional.interpolate(glob_cond, scale_factor=self.img_downsample_factor / self.scaling_factor)

        return torch.cat((glob_cond, rolled_attention), dim=1)


    def sample(self, batch: tuple,
                  save: bool = True, 
                  sample_step: int = 10, 
                  save_path: str = None,
                  ):
        self.log_print(f"Generating sample at epoch {self.epoch} iteration step: {self.iteration_counter}")

        save_path = self.paths['samples'] if save_path is None else save_path
        save_path = os.path.join(save_path, f'sample_{self.epoch}_{self.iteration_counter}_GPU{self.rank}')
        if save and not os.path.exists(save_path):
            os.mkdir(save_path)
        
        hr, lr, lr_up = batch
        if save:
            utils.save_images(hr, os.path.join(save_path, f'original.jpg'), unnormalized=True)
            utils.save_images(lr_up, os.path.join(save_path, f'lowres.jpg'), unnormalized=True)

        batch_size = hr.shape[0]

        x = torch.randn((batch_size, 3, self.image_size, self.image_size), device=self.device)

        for i in reversed(range(0, self.noise_steps)):
            roll_pattern = self.generate_rolling_pattern(self.rolling_pattern)
            inverse_roll_pattern = tuple(-shift for shift in roll_pattern)

            patches = torch.roll(x, shifts=roll_pattern, dims=(-2, -1))
            patches = self.patchify(patches)
            
            # condition = self.generate_rolling_condition(lr_up, x, attention_shift=inverse_roll_pattern, train=False)
            condition = self.generate_rolling_condition(lr, attention_shift=inverse_roll_pattern, train=False)
            patches = self.swin_diffuser.sample_step(patches, i, condition)

            # Undo the rolling
            patches = self.unpatchify(patches)
            x = torch.roll(patches, shifts=inverse_roll_pattern, dims=(-2, -1))

            if i % sample_step == 0 and save:
                utils.save_images(x, os.path.join(save_path, f'res_{i}.jpg'), unnormalized=True)
                sam = self.swin_diffuser.res2img(x, lr_up)
                utils.save_images(sam, os.path.join(save_path, f'sample_{i}.jpg'), unnormalized=True)
        
        return self.swin_diffuser.res2img(x, lr_up)


    def scheduler_update(self):
        for trainer in [self.swin_diffuser]:
            trainer.scheduler.step()

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.measure_batch(pred, target, reset_result=True)
        self.log_print(self.metric.get_result())


class DWT_TrainerGroup(TrainerGroup):
    def __init__(self, cfg,
                 rank, 
                 init_test, 
                 customer_dataloader=None, 
                 train: bool = True, 
                 memroy_check: bool = True) -> None:
        super().__init__(cfg, rank, init_test, customer_dataloader, train, memroy_check)
        assert self.image_size % 2 == 0
        self.res_scale = cfg.group.res_scale

        self.diffuser_kernel_size = self.image_size // 2
        self.xfm = DWTForward(J=1, mode='zero', wave='haar').to(self.device)  # TODO: check the J value 
        self.ifm = DWTInverse(mode='zero', wave='haar').to(self.device)
        self.trainers = {}
        self.losses = {}
        # self.tags = ['LA', 'LV', 'LH', 'LD']
        self.tags = ['LV', 'LH', 'LD']
        for tag in self.tags:
            # beta_start = cfg.group.LL_beta_start if tag == 'LA' else cfg.group.beta_start
            # beta_end = cfg.group.LL_beta_end if tag == 'LA' else cfg.group.beta_end
            # self.trainers[tag] = Trainer(cfg, rank, self.diffuser_kernel_size, num_channels=self.num_channels, device=self.device, train=self.train_mod,
            #                         with_condition=False, beta_start=beta_start, beta_end=beta_end)
            self.trainers[tag] = Trainer(cfg, rank, self.diffuser_kernel_size, num_channels=self.num_channels, device=self.device, train=self.train_mod)
            self.losses[tag]  = 0

        self.metric = Measure(2)

        if len(os.listdir(self.paths['checkpoint_meta'])) != 0:
            self.load_checkpoints(os.path.join(self.paths['checkpoint_meta'], "checkpoint_meta.pth"))
        else:
            self.log_print('No checkpoint found, starting from scratch.')

        if init_test:
            import structure_test
            structure_test.dwt_test(self)
            structure_test.dwt_forward_test(self)

    def save_checkpoints(self):
        checkpoints = {}
        for tag in self.tags:
            checkpoints[tag] = self.trainers[tag].snapshot()
        checkpoints['Iteration'] = self.iteration_counter
        checkpoints['epoch'] = self.epoch
        torch.save(checkpoints, os.path.join(self.paths['checkpoints'],
                                             f"{self.dataset}_{self.epoch}_{self.iteration_counter}.pth"))
        torch.save(checkpoints, os.path.join(self.paths['checkpoint_meta'],
                                             f"checkpoint_meta.pth"))


    def load_checkpoints(self, filename: str):
        checkpoints = torch.load(filename, map_location="cuda")
        self.iteration_counter = checkpoints['Iteration']
        self.epoch = checkpoints['epoch']
        for tag in self.tags:
            self.trainers[tag].load_checkpoint(checkpoints[tag])
        self.log_print(f"Checkpoint loaded from {filename}!")
        self.log_print(f"Current iterarion: {self.iteration_counter}")


    def train(self):
        self.log_print('Starting training procedure...')
        self.print_device_info()
        self.log_print("***************************** Iteration *********************************")
        total_iter_per_batch = len(self.dataloader)
        self.log_print(f'Length of dataloader: {total_iter_per_batch}')
        sample_iter = iter(self.sample_dataloader)

        while True:
            train_iter = iter(self.dataloader)

            while True:
                if self.iteration_counter >= total_iter_per_batch:
                    break
                batch, _, _ = next(train_iter)

                self.train_one_step(batch.to(self.device))

                if self.iteration_counter % self.record_loss_every == 0:
                    lr = self.trainers['LD'].optimizer.param_groups[-1]['lr']
                    self.log_print(f'Epoch {self.epoch} \tIteration: {self.iteration_counter}\tlr: {lr}\nLoss: {self.losses}')

                if (self.iteration_counter) % self.save_every == 0:
                    if self.rank == 0:
                        self.save_checkpoints()
                    batch = next(sample_iter)
                    batch = self.all_to_device(batch)
                    pred = self.sample(batch)
                    self.evaluate(pred, batch)
                
                    if self.world_size > 1:
                        dist.barrier()

                self.iteration_counter += 1

            self.scheduler_update()
            self.log_print(f'Episode {self.epoch} end.')
            self.epoch += 1
            self.iteration_counter = 0

    
    def train_one_step(self, batch: tuple) -> float:
        dwt_result = list(self.apply_dwt(batch))
        condition = dwt_result[0]
        dwt_result = dwt_result[1:]

        for target, tag in zip(dwt_result, self.tags):
            trainer = self.trainers[tag]
            loss = trainer.train_one_step(target, condition)
            if loss is not None:
                self.losses[tag] = loss

        return None
    
    def sample(self, batch:torch.Tensor, save: bool = True):
        self.log_print(f"Generating sample at epoch {self.epoch} iteration step: {self.iteration_counter}")
        ground_truth, _, _ = batch
        ground_truth = self.apply_dwt(ground_truth)

        save_path = self.paths['samples']
        save_path = os.path.join(save_path, f'sample_{self.epoch}_{self.iteration_counter}_GPU{self.rank}')
        if save and not os.path.exists(save_path):
            os.mkdir(save_path)

        utils.save_images(self.combine_channel(*ground_truth), os.path.join(save_path, f'ground_truth.jpg'),unnormalized=True)
        condition = list(ground_truth)[0]
        results = [condition]
        for tag in self.tags:
            result = self.trainers[tag].sample(sample_count=self.eval_batch_size, condition=condition)
            results.append(result)
            
        pred = self.apply_idwt(*tuple(results))
        if save:
            utils.save_images(pred, os.path.join(save_path, f'sample.jpg'),
                                            unnormalized=True)
            utils.save_images(self.combine_channel(*tuple(results)), 
                                os.path.join(save_path, f'dwt_sample.jpg'),
                                            unnormalized=True)
        return pred

    def sample_v2(self, batch:torch.Tensor, save: bool = True, save_step: int = 20):
        ground_truth, _, _ = batch
        ground_truth = self.apply_dwt(ground_truth)

        save_path = './tmp'
        save_path = os.path.join(save_path, f'sample_{self.epoch}_{self.iteration_counter}_GPU{self.rank}')
        if save and not os.path.exists(save_path):
            os.mkdir(save_path)

        utils.save_images(self.combine_channel(*ground_truth), os.path.join(save_path, f'ground_truth.jpg'),unnormalized=True)
        condition = list(ground_truth)[0]
        x = {}
        for tag in self.tags:
            x[tag] = torch.randn((self.eval_batch_size, 3, self.diffuser_kernel_size, self.diffuser_kernel_size), device=self.device)

        for i in reversed(range(0, self.noise_steps)):
            results = [condition]
            for tag in self.tags:
                x[tag] = self.trainers[tag].sample_step(x[tag], i, condition)
                results.append(x[tag])

            if i % 20 == 0:
                utils.save_images(self.combine_channel(*tuple(results)), os.path.join(save_path, f'sample_{i}.jpg'),unnormalized=True)


    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.measure_batch(pred, target, reset_result=True)
        self.log_print(self.metric.get_result())

    def print_basic_info(self):
        self.print_device_info()

    def apply_dwt(self, images):
        images_LL, hfreq_tuple = self.xfm(images)
        return images_LL, hfreq_tuple[0][:, :, 0, :, :], hfreq_tuple[0][:, :, 1, :, :], hfreq_tuple[0][:, :, 2, :, :]

    def apply_idwt(self, LA: torch.Tensor, LV: torch.Tensor, 
                   LH: torch.Tensor, LD: torch.Tensor):

        sr_HFreqs = torch.cat([LV.unsqueeze(2), LH.unsqueeze(2), LD.unsqueeze(2)], 2)
        sr_images = self.ifm((LA, [sr_HFreqs]))
        return sr_images

    def combine_channel(self, LA, LV, LH, LD):
        foo = torch.cat([LA, LV], dim=-1)
        bar = torch.cat([LH, LD], dim=-1)
        return torch.cat([foo, bar], dim=-2)

    def scheduler_update(self):
        for tag in self.tags:
            self.trainers[tag].scheduler.step()

def get_trainergroup(cfg, rank, init_test, train:bool = True) -> TrainerGroup:
    group_type = cfg.group.type
    if group_type == 'Pyramid':
        return Pyramid_TrainerGroup(cfg, rank, init_test, train=train)
    elif group_type == 'Swin':
        return Swin_TrainerGroup(cfg, rank, init_test, train=train)
    elif group_type == 'DWT':
        return DWT_TrainerGroup(cfg, rank, init_test, train=train)
    else:
        raise NotImplementedError(f'{group_type} is not implemented as group type!')
