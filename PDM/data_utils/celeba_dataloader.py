import numpy as np
from tqdm import tqdm
from skimage.io import imsave, imread
from skimage import img_as_float
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

from .imresize import imresize



class celeba_dataset(Dataset):
    def __init__(self, image_dir, scale, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.transform = transform
        self.scale = scale

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img_hr = Image.open(img_path)

        img_hr = Image.fromarray(np.uint8(img_hr))
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / self.scale, method='bilinear')  # np.uint8 [H, W, C]
        img_lr_up = imresize(img_lr / 256, self.scale)  # np.float [H, W, C]
        img_hr, img_lr, img_lr_up = [self.transform(x).float() for x in [img_hr, img_lr, img_lr_up]]

        return img_hr, img_lr, img_lr_up

def get_celeba_dataset(image_dir, scale):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)])
    dataset = celeba_dataset(image_dir, scale=scale, transform=data_transforms)
    return dataset
    

def get_celeba_dataloader(img_dir, batch_size, scale):
    dataset = get_celeba_dataset(img_dir, scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


def group_downsample(work_dir: str, save_dir: str, size: int):
    # work_dir = './CelebA'
    # save_dir = './resize_128'
    img_list = os.listdir(work_dir)
    count = 0
    for img in tqdm(img_list):
        try:
            if not img.endswith('.jpg'):
                print('Not image')
                continue
            new_img_double = resizee_single_img(os.path.join(work_dir, img))
            imsave(os.path.join(save_dir, f'{count}.jpg'), convertDouble2Byte(new_img_double))
            count += 1
        except:
            print(img)
            continue

def resizee_single_img(img_path, size: int):
    img_uint8 = imread(img_path)
    img_double = img_as_float(img_uint8)
    return imresize(img_double, output_shape=(size, size))


def convertDouble2Byte(img: np.ndarray):
    foo = np.clip(img, 0, 1) * 255.
    foo = foo.astype(np.uint8)
    return foo


if __name__ == '__main__':
    dataloader = get_celeba_dataloader('/bigpool/homes/yeyun/projects/PyramidDiffusionModel/datasets/celeba/resize_64', batch_size=16, scale=2)
    data = next(iter(dataloader))
    for i in data:
        print(i.shape)