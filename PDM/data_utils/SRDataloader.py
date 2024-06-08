import numpy as np
from tqdm import tqdm
from skimage.io import imsave, imread
from skimage import img_as_float
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

from .imresize import imresize



class SRDataset(Dataset):
    def __init__(self, image_dir, scale, format: str = 'jpg', 
                 augmentation: bool = True, train: bool = True):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith("." + format)]
        preprocess_transforms = []
        if augmentation and train:
            preprocess_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ]
        self.preprocess = transforms.Compose(preprocess_transforms)
        self.postprocess_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])
        self.scale = scale
        test_img_size = 5000
        if train:
            self.image_files = self.image_files[:-test_img_size]
        else:
            self.image_files = self.image_files[-test_img_size:]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img_hr = Image.open(img_path)

        img_hr = Image.fromarray(np.uint8(img_hr))
        img_hr = self.preprocess(img_hr)
        img_hr = np.asarray(img_hr).copy()  # maybe solved the warning problem
        img_lr = imresize(img_hr, 1 / self.scale, method='bilinear')  # np.uint8 [H, W, C]
        img_lr_up = imresize(img_lr / 255, self.scale)  # np.float [H, W, C]
        img_hr, img_lr, img_lr_up = [self.postprocess_transforms(x).float() for x in [img_hr, img_lr, img_lr_up]]

        return img_hr, img_lr, img_lr_up

def get_celeba_dataset(image_dir, scale, train, augmentation):
    dataset = SRDataset(image_dir, scale=scale, format='jpg', augmentation=augmentation, train=train)
    return dataset
    
def get_celeba_dataloader(img_dir, batch_size, val_batch_size, scale, augmentation: bool = False):
    train_dataset = get_celeba_dataset(img_dir, scale, train=True, augmentation=augmentation)
    val_dataset = get_celeba_dataset(img_dir, scale, train=False, augmentation=augmentation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True)
    return train_loader, val_loader


if __name__ == '__main__':
    dataloader = get_celeba_dataloader('/bigpool/homes/yeyun/projects/PyramidDiffusionModel/datasets/celeba/resize_64', batch_size=16, scale=2)
    data = next(iter(dataloader))
    for i in data:
        print(i.shape)