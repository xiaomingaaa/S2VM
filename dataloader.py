import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

class PretrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.samples = args.scale
        self.range_data = range(1, args.scale+1)

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        
    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        d1 = index+1
        d2 = random.choice(self.range_data)

        img1 = Image.open(f'datasets/pretrain/img/{d1}.png')
        img2 = Image.open(f'datasets/pretrain/img/{d2}.png')
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2

class MAEDDIDataset(Dataset):
    def __init__(self, data_name='Deng', data_type='training', fewshot=None):
        super().__init__()
        drug_set = []
        with open('datasets/Deng_dataset/drug_smiles.csv', 'r') as f:
            f.readline()
            bar = tqdm(f)
            for idx, line in enumerate(bar):
                did, smiles = line.strip().split(',')
                drug_set.append(did)

        self.samples = []
        if not fewshot:
            path = f'datasets/{data_name}_dataset/ddi_{data_type}1.csv'
        else:
            path = f'datasets/{data_name}_dataset/{fewshot}/ddi_{data_type}1.csv'
        print(path)
        
        self.samples = []
        with open(path, 'r') as f:
            f.readline()
            bar = tqdm(f)
            for idx, line in enumerate(bar):
                h, label, t = line.strip().split(',')
                label = int(label)


                self.samples.append([f'datasets/drug_image/{h}.png', f'datasets/drug_image/{t}.png', label])

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        d1, d2, label = self.samples[index]
        img1 = Image.open(d1)
        img2 = Image.open(d2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, label
    
class TSDDIDataset(Dataset):
    def __init__(self, fold='S1', data_type='train' ):
        super().__init__()
        self.samples = []
        path = f'datasets/TWOSIDES/{fold}/{data_type}_ddi.txt'
        print(path)
        self.samples = []
        with open(path, 'r') as f:
            f.readline()
            bar = tqdm(f)
            for idx, line in enumerate(bar):
                h, t, label, bin_label = line.strip().split('\t')
                self.samples.append([f'datasets/drug_images/{h}.png', f'datasets/drug_images/{t}.png', int(bin_label)])

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        d1, d2, label = self.samples[index]
        img1 = Image.open(d1)
        img2 = Image.open(d2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, label
