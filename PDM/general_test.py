from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

# class foobar(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.ll = nn.Linear(10, 10)

#     def forward(self, x):
#         return self.ll(x)


# num_epochs = 100
# number_warmup_epochs = 50


# model = foobar()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# train_scheduler = CosineAnnealingLR(optimizer, num_epochs)

# train_scheduler.step()
# train_scheduler.step()
# train_scheduler.step()
# train_scheduler.step()
# train_scheduler.step()

# warmup = lambda current_step: current_step / number_warmup_epochs
# warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)

# scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [number_warmup_epochs])

# for i in range(num_epochs):
#     lr = optimizer.param_groups[-1]['lr']
#     print(f'lr for epidoe {scheduler.last_epoch}: {lr}')
#     scheduler.step()

import hydra
from hydra.core.hydra_config import HydraConfig

from ddpm_group import Swin_TrainerGroup

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir 
    cfg.work_dir = work_dir


    tg = Swin_TrainerGroup(cfg, rank=-1,init_test=False)
    data, _ = next(iter(tg.dataloader))
    tg.sample_sr_rolling(data)


if __name__ == '__main__':
    main()