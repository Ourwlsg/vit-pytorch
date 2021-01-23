import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random
import cv2
import albumentations as A
import torchvision.transforms as transforms
from lib.rand_augment import RandomAugment

from cfg.cfg import cfg

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


def get_train_transform_strong(mean=cfg.mean, std=cfg.std, size=512):
    train_transform = A.Compose([
        A.Resize(width=int(size * (256 / 224)), height=int(size * (256 / 224)), interpolation=cv2.INTER_LINEAR),
        A.RandomCrop(width=size, height=size),
        # A.Transpose(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
            A.ElasticTransform(p=0.3),
        ], p=0.1),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(p=0.3),
            A.IAAEmboss(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ], p=0.1),
        A.GaussianBlur(p=0.2),
        A.RandomGamma(p=0.1),
        A.CoarseDropout(p=0.5),
        # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(p=0.1),
        # A.ToTensor(),
        A.Normalize(mean=mean, std=std),
    ])
    return train_transform


def get_train_transform(mean=cfg.mean, std=cfg.std, size=512):
    train_transform = A.Compose([
        A.Resize(width=int(size * (256 / 224)), height=int(size * (256 / 224)), interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(width=size, height=size),
        # A.CoarseDropout(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.1),
        A.ShiftScaleRotate(p=0.3),
        A.Normalize(mean=mean, std=std),
    ])
    return train_transform


def get_test_transform(mean=cfg.mean, std=cfg.std, size=512):
    return A.Compose([
        A.Resize(width=int(size * (256 / 224)), height=int(size * (256 / 224)), interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(width=size, height=size),
        # A.ToTensor(),
        A.Normalize(mean=mean, std=std),
    ])


class CassavaDataset(Dataset):
    def __init__(self, label_file, mode, transform_name):
        with open(label_file, 'r') as f:
            self.cls_num = cfg.CLS_NUM
            # label_file的格式, (label_file image_label)
            dataAndtarget = np.array(list(map(lambda line: line.strip().split(' '), f)))
            self.data = dataAndtarget[:, 0]
            self.targets = [int(target) for target in dataAndtarget[:, 1]]

            train = True if mode == "train" else False
            self.train = train
            self.transform_name = transform_name
        if self.train:
            self.augment = RandomAugment(N=3, M=7)
            rand_number = cfg.SEED
            np.random.seed(rand_number)
            random.seed(rand_number)
            if self.transform_name == 'RandomAugment':
                self.transform = get_test_transform(size=cfg.INPUT_SIZE)
            else:
                self.transform = get_train_transform(size=cfg.INPUT_SIZE)
        else:
            self.transform = get_test_transform(size=cfg.INPUT_SIZE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            if self.transform_name == 'RandomAugment':
                img = self.augment(data=img)["data"]
            img = self.transform(image=img)['image']

        return torch.from_numpy(img).permute(2, 0, 1), target

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos


if __name__ == '__main__':
    train_label_file = "/home/zhucc/kaggle/pytorch_classification/data/cv/fold_0/train.txt"
    val_label_file = "/home/zhucc/kaggle/pytorch_classification/data/cv/fold_0/val.txt"

    train_set = CassavaDataset(train_label_file, mode="train", transform_name="RandomAugment")

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    # val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)
    #
    train_loader = iter(train_dataloader)
    train_data, train_label = next(train_loader)
    print(train_data.shape, train_label)
    # print("================================")
    # val_loader = iter(val_dataloader)
    # val_data, val_label, val_meta = next(val_loader)
    # print(val_data.shape, val_label, val_meta)
