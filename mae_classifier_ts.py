import os
import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import TSDDIDataset
import warnings
warnings.filterwarnings("ignore")
from mae_model import *
from utils import setup_seed, eval_mae_loader_ts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default='ckpts/mae/vit-t-mae_8layers_patch16.pt')
    parser.add_argument('--dataset', type=str, default='ts',
                        help='[Deng, Ryu, drugbank_ind]')
    parser.add_argument('--fold', type=str, default='S1',
                        help='[S1, S2]')

    args = parser.parse_args()
    args.output_model_path = f'ckpts/mae/vit-t-classifier-from_scratch_8layers_{args.dataset}_{args.fold}.pt'

    if args.dataset == 'Deng':
        args.num_classes = 65
    elif args.dataset == 'ts':
        args.num_classes = 1

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = TSDDIDataset(args.fold, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)

    val_dataset = TSDDIDataset(args.fold, 'valid')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=True, num_workers=4)

    test_dataset = TSDDIDataset(args.fold, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, load_batch_size, shuffle=True, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        writer = SummaryWriter(os.path.join('logs', args.dataset, 'pretrain-cls'))
    else:
        print('No pretrain')
        model = MAE_ViT()
        writer = SummaryWriter(os.path.join('logs', args.dataset, 'scratch-cls'))
    model = torch.load('ckpts/mae/vit-t-classifier-from_scratch_8layers_ts_s1_frozen.pt',  map_location='cpu').cuda()

    for name, param in model.named_parameters():
        if "transformer" in name:
            print(name)
            param.requires_grad = False

    loss_fn = torch.binary_cross_entropy_with_logits

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img1, img2, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img1 = img1.to(device)
            img2 = img2.to(device)

            label = label.to(device)
            logits = model(img1, img2)
            loss = loss_fn(logits.squeeze(), label.float()).mean()
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())

        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average training loss is {avg_train_loss}')

        acc, auc, aupr = eval_mae_loader_ts(model, val_dataloader, device)  

        if auc > best_val_acc:
            best_val_acc = auc
            print(f'saving best model with auc {best_val_acc} at {e} epoch!')   
            print(acc, auc, aupr)    
            torch.save(model, args.output_model_path)

        writer.add_scalar('cls/auc', auc, global_step=e)
        writer.add_scalar('cls/aupr', aupr, global_step=e)

    model = torch.load(args.output_model_path, map_location='cpu')
    print(eval_mae_loader_ts(model, test_dataloader, device))