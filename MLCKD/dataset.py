import os
from torchvision import transforms, datasets
import torch.utils.data as data
import torch
from torchvision.datasets import STL10
from typing import Any
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import argparse
from PIL import Image

def create_loader(batch_size, data_dir, data):
    data_dir = os.path.join(data_dir,data)
    if data == 'CIFAR100':
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        num_classes = 100
        image_size = 32

        return train_loader, test_loader, num_classes, image_size

    if data.lower() == 'cub_200_2011':
        n_class = 200
    elif data.lower() == 'dogs':
        n_class = 120
    elif data.lower() == 'mit67':
        n_class = 67
    elif data.lower() == 'stanford40':
        n_class = 40
    elif data == "STL-10":
        n_class=10
    else:
        n_class = 1000
    if data == "STL-10":
        image_size = 32
        transform_train = transforms.Compose(
            [transforms.Resize((32,32)),transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),])
        transform_test = transforms.Compose(
            [transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),])
        trainset = datasets.STL10(root=data_dir,
                              split="train",
                              transform=transform_train,
                              download=False)
        testset = datasets.STL10(root=data_dir,
                              split="test",
                              transform=transform_train,
                              download=False)
    elif data == "tiny-imagenet-200":
        image_size = 32
        train_loader, test_loader, n_class =  load_tinyimagenet(data_dir,batch_size)
        return train_loader, test_loader, n_class, image_size
    else:
        image_size = 224
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

        transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

        trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)

    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              num_workers=2)
    return train_loader, test_loader, n_class, image_size

 
class TrainTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None) -> None:
        super().__init__()
        self.filenames = glob.glob(root + "/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
def load_tinyimagenet(root,batch_size):
    nw = 4
    id_dic = {}
    for i, line in enumerate(open(root+'/wnids.txt','r')):
        id_dic[line.replace('\n', '')] = i
    num_classes = len(id_dic)
    data_transform = {
        "train": transforms.Compose([transforms.Resize((32,32)),
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     ]),
        "val": transforms.Compose([transforms.Resize((32,32)),
                                   transforms.ToTensor(),
                                   ])}
    train_dataset = TrainTinyImageNet(root, id=id_dic, transform=data_transform["train"])
    val_dataset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])
 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    
    print("TinyImageNet Loading SUCCESS"+
          "\nlen of train dataset: "+str(len(train_dataset))+
          "\nlen of val dataset: "+str(len(val_dataset)))
    
    return train_loader, val_loader, num_classes