from utils import loadSmilesAndSave
from tqdm import tqdm
import argparse

def process_pretrain_images():
    with open('datasets/pretrain/data.csv', 'r') as f:
        f.readline()
        for line in f:
            infos = line.strip().split(',')
            index, smiles = infos[0], infos[-1]
            path = f'datasets/pretrain/img/{index}.png'
            loadSmilesAndSave(smiles, path)

def process_mae_images_pre():
    with open('datasets/Deng_dataset/drug_smiles.csv', 'r') as f:
        f.readline()
        for idx, line in enumerate(f):
            did, smiles = line.strip().split(',')
            path = f'datasets/drug_images/{did}.png'
            loadSmilesAndSave(smiles, path)

def process_ts():
    import json
    with open('datasets/TWOSIDES/id2drug.json', 'r') as f:
        drugs = json.load(f)
        for d in drugs:
            print(d)
            smiles = drugs[d]['smiles']
            path = f'datasets/drug_images/{d}.png'
            loadSmilesAndSave(smiles, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="DDI", help="[pretrain, DDI, twosides_ind]")

    args = parser.parse_args()

    if args.type == 'DDI':
        process_mae_images_pre()
    elif args.type == 'pretrain':
        process_pretrain_images()
    elif args.type == 'twosides_ind':
        process_ts()
    else:
        raise Exception(f'Can not process {args.type}!')