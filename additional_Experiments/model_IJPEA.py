import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from x_transformers import Encoder, Decoder
import copy

'''
PatchEmbed class, adapted from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632 I think, but I dont have medium premium so idk
- This class is used to convert the image into patches using a convolutional layer
'''
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size
        #calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        #convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        

    def forward(self, x):
        x = self.conv(x)
        #flatten the patches
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x

'''Lightweight Predictor Module using VIT to predict target patches from context patches'''
class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        
        self.predictor = Decoder(dim = embed_dim, depth = depth, heads = num_heads)
    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim = 1)
        x = self.predictor(x)
        #return last len(target_masks) tokens
        l = x.shape[1]
        return x[:, l - target_masks.shape[1]:, :]
    

class IJEPA_DualImages_Full(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, enc_depth, pred_depth, num_heads,
                 post_emb_norm=False, M=4, mode='train', layer_dropout=0., device='cuda'):
        super().__init__()
        self.M = M
        self.mode = mode
        self.layer_dropout = layer_dropout

        #define the patch embedding and positional embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_dim  = (self.patch_embed.patch_shape[0], self.patch_embed.patch_shape[1])
        self.num_tokens = self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        #define the cls and mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        #define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.teacher_encoder = Encoder(
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth, 
            layer_dropout=self.layer_dropout,
        )  
        self.student_encoder = copy.deepcopy(self.teacher_encoder).cuda()
        self.predictor = Predictor(embed_dim, num_heads, pred_depth)


    @torch.no_grad() 
    def get_target_block(self, target_encoder, x, patch_dim, aspect_ratio, scale, M):  
        #get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x)
        x = self.norm(x)
        #get the patch dimensions
        patch_h, patch_w = patch_dim
        #get the number of patches
        num_patches = patch_h * patch_w
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the patches in the target block
        target_block = torch.zeros((M, x.shape[0], block_h*block_w, x.shape[2]))
        target_patches = []
        all_patches = []
        for z in range(M):
            #get the starting patch
            start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            patches = []
            #get the patches in the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)
                    
            #get the target block
            target_patches.append(patches)
            target_block[z] = x[:, patches, :]
        return target_block.cuda(), target_patches, all_patches

    def get_context_block(self, x, patch_dim, aspect_ratio, scale, target_patches):
        patch_h, patch_w = patch_dim
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        #get the patches in the context_block
        patches = []
        for i in range(block_h):
            for j in range(block_w):
                if start_patch + i * patch_w + j not in target_patches: #remove the target patches
                    patches.append(start_patch + i * patch_w + j)
        return x[:, patches, :]

    def forward(self, img1, img2, target_aspect_ratio=None, target_scale=None, context_aspect_ratio=None, context_scale=None):
        # patch embed
        x1 = self.patch_embed(img1)  # [B, N, E]
        x2 = self.patch_embed(img2)  # [B, N, E]

        # add positional embedding
        x1 = x1 + self.pos_embedding
        x2 = x2 + self.pos_embedding

        x1 = self.post_emb_norm(x1)
        x2 = self.post_emb_norm(x2)

        if self.mode == 'test':
            x_concat = torch.cat([x1, x2], dim=1)
            return self.student_encoder(x_concat)

        # select target blocks from teacher_encoded of img1
        target_blocks1, target_patches1, all_patches1 = self.get_target_block(
        self.teacher_encoder, x1, self.patch_dim, target_aspect_ratio, target_scale, self.M
    )
        target_blocks2, target_patches2, all_patches2 = self.get_target_block(
        self.teacher_encoder, x2, self.patch_dim, target_aspect_ratio, target_scale, self.M
    )
        m, b, n, e = target_blocks1.shape


        context_block1 = self.get_context_block(x1, self.patch_dim, context_aspect_ratio, context_scale, all_patches1)
        context_block2 = self.get_context_block(x2, self.patch_dim, context_aspect_ratio, context_scale, all_patches2)

        context_concat = torch.cat([context_block1, context_block2], dim=1)

        context_encoding = self.student_encoder(context_concat)
        context_encoding = self.norm(context_encoding)

        
        prediction_blocks1 = torch.zeros((m, b, n, e), device=x1.device)
        prediction_blocks2 = torch.zeros((m, b, n, e), device=x1.device)

        for i in range(m):
            # 
            mask1 = self.mask_token.repeat(b, n, 1)
            pos_emb1 = self.pos_embedding[:, target_patches1[i], :]
            target_mask1 = mask1 + pos_emb1
            prediction_blocks1[i] = self.predictor(context_encoding, target_mask1)

            # 
            mask2 = self.mask_token.repeat(b, n, 1)
            pos_emb2 = self.pos_embedding[:, target_patches2[i], :]
            target_mask2 = mask2 + pos_emb2
            prediction_blocks2[i] = self.predictor(context_encoding, target_mask2)

        return prediction_blocks1, target_blocks1, prediction_blocks2, target_blocks2
    

class IJPEA_Classifer(nn.Module):
    def __init__(self, encoder : IJEPA_DualImages_Full, num_classes=65) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(encoder.student_encoder.dim, num_classes)

    
    def forward(self,img1,img2):
        feat = self.encoder(img1, img2)  #  [B, L, C]
        pooled = feat.mean(dim=1)
        return self.head(pooled)