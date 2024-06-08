import numpy as np
import lpips
import torch
from data_utils.imresize import imresize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class Measure:
    def __init__(self, scaling_factor, net='alex'):
        self.metric_keys = ['psnr', 'ssim', 'lpips', 'lr_psnr']
        self.model = lpips.LPIPS(net=net)
        self.reset_result()
        self.num_samples = 0
        self.scaling_factor = scaling_factor
    
    def reset_result(self):
        self.result = {k: 0 for k in self.metric_keys}
        self.num_samples = 0

    def measure_batch(self, pred, target_batch, reset_result:bool=False):
        if reset_result:
            self.reset_result()
        
        for b in range(pred.shape[0]):
            self.measure(pred[b], target_batch[0][b], 
                         target_batch[1][b], self.scaling_factor)

    def measure(self, imgA, imgB, img_lr, sr_scale):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale:

        Returns: dict of metrics

        """
        if isinstance(imgA, torch.Tensor):
            imgA = np.round((imgA.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            imgB = np.round((imgB.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            img_lr = np.round((img_lr.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
        imgA = imgA.transpose(1, 2, 0)
        imgA_lr = imresize(imgA, 1 / sr_scale)
        imgB = imgB.transpose(1, 2, 0)
        img_lr = img_lr.transpose(1, 2, 0)
        psnr = self.psnr(imgA, imgB)
        ssim = self.ssim(imgA, imgB)
        lpips = self.lpips(imgA, imgB)

        lr_psnr = self.psnr(imgA_lr, img_lr)
        res = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'lr_psnr': lr_psnr}

        self.result['psnr'] += psnr
        self.result['ssim'] += ssim
        self.result['lpips'] += lpips
        self.result['lr_psnr'] += lr_psnr
        self.num_samples += 1

    def get_result(self):
        return {k: float(v / self.num_samples) for k, v in self.result.items()}
    

    def lpips(self, imgA, imgB):
        device = next(self.model.parameters()).device
        tA = t(imgA).to(device)
        tB = t(imgB).to(device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        score, _ = ssim(imgA, imgB, full=True, 
                        multichannel=True, 
                        data_range=255, channel_axis=-1)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1