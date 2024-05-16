from abc import abstractmethod, ABC
import torch
from torch import nn


class Downsampler(ABC):
    def __init__(self, num_down: int, factor, device) -> None:
        self.num_down = num_down
        self.factor = factor
        self.device = device

    @ abstractmethod
    def transform(self, img: torch.Tensor):
        """
        Transform an image into a bunch of images in different scale
        Input:
        img: torch tensor with shape(bs, c, width, height)
        """
        pass
        
    @ abstractmethod
    def upsample(self, img: torch.Tensor):
        """ Upsample the input tensor with the given scale factor. For backward sampling.

        Args:
            img (torch.Tensor): with shape (bs, c, w, h)
        """
        pass


class LaplacianDownsampler(Downsampler):
    def __init__(self, num_down, factor:int=2, device='cpu') -> None:
        super().__init__(num_down, factor, device)
        self.avg_pool = nn.AvgPool2d(kernel_size=factor, stride=factor)
    
    @torch.no_grad()
    def _downsample(self, img: torch.Tensor):
        with torch.no_grad():
            downsampled_output = self.avg_pool(img)
        return downsampled_output

    @torch.no_grad()
    def upsample(self, img: torch.Tensor):
        with torch.no_grad():
            upsampled_output = nn.functional.interpolate(img, scale_factor=self.factor, mode='nearest')
        return upsampled_output

    @torch.no_grad()
    def transform(self, img: torch.Tensor):
        tmp_img = img.clone().to(self.device)
        gaussian_pyr = []
        laplacian_pyr = []
        fid_pyr = [img.clone()]

        for _ in range(self.num_down):
            blur_img = self._downsample(tmp_img)
            blur_img_org_size = self.upsample(blur_img)
            gaussian_pyr.append(blur_img_org_size)
            laplacian_pyr.append(tmp_img - blur_img_org_size)
            fid_pyr.append(blur_img)
            tmp_img = blur_img

        return gaussian_pyr, laplacian_pyr, fid_pyr[:-1]


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # from torchvision.datasets import CIFAR10
    # from torchvision import transforms
    # import matplotlib.pyplot as plt
    
    # batch_size = 16
    # cifar = CIFAR10(root='../datasets', train=True, download=True, transform=transforms.ToTensor())
    # dataLoader = DataLoader(cifar, batch_size=batch_size, shuffle=True)
    # batch, _ = next(iter(dataLoader))
    ds = LaplacianDownsampler(num_down=4, factor=2)
    # gau, lap = ds.transform(batch)

    # for num, img in enumerate(gau):
    #     fig = plt.figure()
    #     tmp = img[0].permute(1, 2, 0).detach()
    #     tmp = (tmp * 255.0).type(torch.uint8)
    #     print(tmp[:, :, 0])
    #     plt.imshow(tmp, interpolation='nearest')
    #     plt.savefig(f'./gau_{num}.jpg')
    #     plt.close()
    # for num, img in enumerate(lap):
    #     fig = plt.figure()
    #     tmp = img[0].permute(1, 2, 0).detach()
    #     tmp = (tmp * 255.0).type(torch.uint8)
    #     plt.imshow(tmp, interpolation='nearest')
    #     plt.savefig(f'./lap_{num}.jpg')
    #     plt.close()