import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import ImageEnhance
import random
import numpy as np


def randomFlip(img, depth, gt):
    flip_flag1 = random.randint(0, 1)
    flip_flag2 = random.randint(2, 3)
    if flip_flag1 == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_flag2 == 2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
    return img, depth, gt


def randomRotation(image, depth, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return image, depth, gt

def randomCrop(image, depth, gt):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), depth.crop(random_region), gt.crop(random_region)

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, depth_root, gt_root, trainsize):
        """
        Args:
            image_root: the path of rgb training images.
            depth_root: the path of depth training images.
            gt_root: the path of the corresponding ground truth of training images.
            trainsize: the image size of training images.
            
        """
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        image, depth, gt = randomFlip(image, depth, gt)
        image, depth, gt = randomRotation(image, depth, gt)
        # multiScale
        scale_flag = random.randint(0, 2)
        if scale_flag == 1:
            self.trainsize = 128
        elif scale_flag == 2:
            self.trainsize = 256
        else:
            self.trainsize = 352
        image = self.img_transform(image)
        depth = self.depths_transform(depth)
        gt = self.gt_transform(gt)
        return image, depth, gt

    def filter_files(self):
        assert len(self.images) == len(self.depths) == len(self.gts)
        images = []
        depths = []
        gts = []
        for img_path, depth_path, gt_path in zip(self.images, self.depths, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
        self.images = images
        self.depths = depths
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, depth, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), depth.resize((w, h), Image.NEAREST), \
                   gt.resize((w, h), Image.NEAREST)
        else:
            return img, depth, gt

    def __len__(self):
        return self.size


def get_loader(image_root, depth_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = SalObjDataset(image_root, depth_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, depth_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depth = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')
                      or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        depth = self.binary_loader(self.depth[self.index])
        depth = self.depth_transform(depth).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
