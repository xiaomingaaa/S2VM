import os
import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import PretrainDataset
from model_IJPEA import *
from utils import setup_seed
from torchvision import transforms
import numpy as np
unloader = transforms.ToPILImage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--scale', type=int, default=100000)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--model_path', type=str, default='ckpts/IJEPA/vit-t-mae_8layers_patch16.pt')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    writer = SummaryWriter(os.path.join('logs', 'drug pair', f'mae-pretrain_{args.scale}'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = PretrainDataset(args)
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)

    val_dataset = PretrainDataset(args)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)
    model = IJEPA_DualImages_Full(img_size=224, patch_size=16, in_chans=3, embed_dim=192,
                       enc_depth=8, pred_depth=4, num_heads=3, mode='train').to(device)
    # model:  = MAE_ViT(patch_size=args.patch_size, mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    loss_fn = nn.MSELoss()
    args.target_aspect_ratio = (0.75,1.5)
    args.target_scale = (0.15, .2)
    args.context_aspect_ratio=1
    args.context_scale=(0.85,1.0)
    args.m=0.996
    args.m_start_end = (.996, 1.)

    step_count = 0
    optim.zero_grad()
    early_stop = 0
    best_loss = 10000
    for e in range(args.total_epoch):
        print(lr_scheduler.get_last_lr())
        model.train()
        early_stop += 1
        if early_stop>=20:
            break

        losses = []
        for img1,img2 in tqdm(iter(dataloader)):
            step_count += 1
            img1 = img1.to(device)
            img2 = img2.to(device)
            target_aspect_ratio = np.random.uniform(args.target_aspect_ratio[0], args.target_aspect_ratio[1])
            target_scale = np.random.uniform(args.target_scale[0], args.target_scale[1])
            context_aspect_ratio = args.context_aspect_ratio
            context_scale = np.random.uniform(args.context_scale[0], args.context_scale[1])
            predicted_img1, ground_true1,predicted_img2, ground_true2, = model(img1,img2,target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
            loss = loss_fn(predicted_img1,ground_true1)+loss_fn(predicted_img2, ground_true2)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')
        ''' visualize the first 16 predicted images on val dataset'''
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop = 0
            torch.save(model.state_dict(), args.model_path)