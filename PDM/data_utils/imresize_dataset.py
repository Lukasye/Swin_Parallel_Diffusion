import os
import numpy as np
from skimage.io import imsave, imread
from skimage import img_as_float
from tqdm import tqdm
import click

from imresize import imresize


@click.command()
@click.option('--size', default=128, help='size of output image')
@click.option('--data_path', default='./datasets/CelebA', help='where is your dataset')
@click.option('--save_path', default='./datasets/celeba', help='where the data saved to')
def main(size, data_path, save_path):
    work_dir = data_path
    save_dir = os.path.join(save_path, 'resize_{size}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img_list = os.listdir(work_dir)
    count = 0
    for img in tqdm(img_list):
        try:
            if not img.endswith('.jpg'):
                print('Not image')
                continue
            img_uint8 = imread(os.path.join(work_dir, img))
            img_double = img_as_float(img_uint8)
            new_img_double = imresize(img_double, output_shape=(size, size))
            imsave(os.path.join(save_dir, f'{count}.jpg'), convertDouble2Byte(new_img_double))
            count += 1
        except:
            print(img)
            continue
        

def convertDouble2Byte(img: np.ndarray):
    foo = np.clip(img, 0, 1) * 255.
    foo = foo.astype(np.uint8)
    return foo
    

if __name__ == '__main__':
    main()
