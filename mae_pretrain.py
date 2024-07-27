import os
import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import PretrainDataset
from mae_model import *
from utils import setup_seed
from torchvision import transforms
unloader = transforms.ToPILImage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--scale', type=int, default=100000)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='ckpts/mae/vit-t-mae_8layers_patch8.pt')

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

    model = MAE_ViT(patch_size=args.patch_size, mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    early_stop = 0
    best_loss = 10000
    for e in range(args.total_epoch):
        model.train()
        early_stop += 1
        if early_stop>=20:
            break

        losses = []
        for img1, img2 in tqdm(iter(dataloader)):
            step_count += 1
            img1 = img1.to(device)
            img2 = img2.to(device)
            predicted_img1, mask1, predicted_img2, mask2 = model(img1, img2)
            loss = torch.mean((predicted_img1 - img1) ** 2 * mask1) / args.mask_ratio + torch.mean((predicted_img2 - img2) ** 2 * mask2) / args.mask_ratio
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
            if e%1 == 0:
                model.eval()
                with torch.no_grad():
                    val_img1, val_img2 = next(iter(val_dataloader))

                    val_img1 = val_img1.to(device)
                    val_img2 = val_img2.to(device)
                    predicted_val_img1, mask1, predicted_val_img2, mask2 = model(val_img1, val_img2)
                    predicted_val_img1 = predicted_val_img1 * mask1 + val_img1 * (1 - mask1)
                    mask_img1 = val_img1 * (1 - mask1)
                    mask_img2 = val_img2 * (1 - mask2)

                    original = mask_img1+mask_img2
                    original = unloader(original.squeeze(0))
                    original.save(f'image_recover/{e}_oring.png', 'PNG', quality = 99)

                    mask_img1 = unloader(mask_img1.squeeze(0))
                    mask_img1.save(f'image_recover/{e}_a.png', 'PNG', quality = 99)

                    predicted_val_img1 = unloader(predicted_val_img1.squeeze(0))
                    predicted_val_img1.save(f'image_recover/{e}_a1.png', 'PNG', quality = 99)

                    mask_img2 = unloader(mask_img2.squeeze(0))
                    mask_img2.save(f'image_recover/{e}_b.png', 'PNG', quality = 99)

                    predicted_val_img2 = unloader(predicted_val_img2.squeeze(0))
                    predicted_val_img2.save(f'image_recover/{e}_b1.png', 'PNG', quality = 99)

            ''' save model '''
            torch.save(model, args.model_path)