from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score,average_precision_score
from dataloader import DDIDataset, ImageMolDDI
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import torch.nn as nn
import os
import torchvision
from PIL import Image
import torch
import random


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loadSmilesAndSave(smis, path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png

        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================

    '''
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
    img.save(path)


    

def evaluate_multi_class(y_pred, labels):
    y_pred = np.argmax(y_pred, axis=1)
    f1 = f1_score(labels, y_pred, average='macro')

    recall = recall_score(labels, y_pred, average='macro')
    precision = precision_score(labels, y_pred, average='macro')
    acc = accuracy_score(labels, y_pred)
    return f1, recall, precision, acc

def evaluate_multi_label(y_pred, labels):
    y_pred = y_pred
    labels = labels
    auc_score = roc_auc_score(labels, y_pred)
    
    aupr_score = average_precision_score(labels, y_pred)
    
    y_label = [1 if i > 0.5 else 0 for i in y_pred]
    acc = accuracy_score(labels, y_label)
    
    return acc, auc_score, aupr_score

def eval_mae_loader(model, loader, device):
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        img1, img2, label = batch
        img1 = img1.to(device)
        img2 = img2.to(device)

        label = label.to(device)
        with torch.no_grad():
            pred = model(img1, img2)
            pred = torch.softmax(pred, dim=1)
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(label.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))
    return evaluate_multi_class(np.array(Y_pre), np.array(Y_true))

def eval_mae_loader_ts(model, loader, device):
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        img1, img2, label = batch
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model(img1, img2)
            pred = torch.sigmoid(pred)
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(label.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))
    return evaluate_multi_label(np.array(Y_pre), np.array(Y_true))
