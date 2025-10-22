import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
# from layers import Gating

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape ## patch_num, 2, 192
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes, remain_T

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img1, img2):
        patches1 = self.patchify(img1)
        patches1 = rearrange(patches1, 'b c h w -> (h w) b c')

        patches2 = self.patchify(img2)
        patches2 = rearrange(patches2, 'b c h w -> (h w) b c')
        
        patches1 = patches1 + self.pos_embedding
        patches2 = patches2 + self.pos_embedding

        patches1, forward_indexes1, backward_indexes1, remain_T = self.shuffle(patches1)
        patches2 = take_indexes(patches2, forward_indexes1)

        patches2 = patches2[remain_T:]
        patches = torch.cat([patches1, patches2], dim=0)
        # patches2, forward_indexes2, backward_indexes2 = self.shuffle(patches2)


        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes1

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # two head
        self.head1 = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.head2 = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)

        # one head
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0] // 2
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature
        # for two head
        # patches1 = self.head1(features)
        # mask1 = torch.zeros_like(patches1)
        # mask1[T-1:] = 1
        # mask1 = take_indexes(mask1, backward_indexes[1:] - 1)
        # img1 = self.patch2img(patches1)
        # mask1 = self.patch2img(mask1)
        
        # patches2 = self.head2(features)
        # mask2 = torch.zeros_like(patches2)
        # mask2[:T+1] = 1
        # mask2 = take_indexes(mask2, backward_indexes[1:] - 1)
        # img2 = self.patch2img(patches2)
        # mask2 = self.patch2img(mask2)
        
        # for one head
        patches = self.head(features)

        # img1
        mask1 = torch.zeros_like(patches)
        mask1[T-1:] = 1
        mask1 = take_indexes(mask1, backward_indexes[1:] - 1)
        img1 = self.patch2img(patches * mask1)
        mask1 = self.patch2img(mask1)

        # img2
        mask2 = torch.zeros_like(patches)
        mask2[:T+1] = 1
        mask2 = take_indexes(mask2, backward_indexes[1:] - 1)
        img2 = self.patch2img(patches * mask2)
        mask2 = self.patch2img(mask2)

        return img1, mask1, img2, mask2

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=192,
                 encoder_layer=8,
                 encoder_head=3,
                 decoder_layer=2,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img1, img2):
        features, backward_indexes = self.encoder(img1, img2)
        predicted_img1, mask1, predicted_img2, mask2 = self.decoder(features,  backward_indexes)
        return predicted_img1, mask1, predicted_img2, mask2

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=65) -> None:
        super().__init__()
        print('Unifying Model')
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.shuffle = encoder.shuffle
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)


    def forward(self, img1, img2):
        patches1 = self.patchify(img1)
        patches2 = self.patchify(img2)

        patches1 = rearrange(patches1, 'b c h w -> (h w) b c')
        patches2 = rearrange(patches2, 'b c h w -> (h w) b c')

        patches1 = patches1 + self.pos_embedding
        patches2 = patches2 + self.pos_embedding

        patches = torch.cat([patches1, patches2], dim=0)

        #### random position perturbation
        # a=a[:,torch.randperm(a.size(1)),:]
#         patches1, forward_indexes1, backward_indexes1, remain_T = self.shuffle(patches1)
#         patches2 = take_indexes(patches2, forward_indexes1)

#         patches2 = patches2[remain_T:]
#         patches = torch.cat([patches1, patches2], dim=0)

        #### position transition
        # patches = torch.cat([patches2, patches1], dim=0)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        logits = self.head(features[0])
        return logits

class ViT_Classifier_Sep(torch.nn.Module):
    '''
    Separate encoders
    '''
    def __init__(self, encoder : MAE_Encoder, num_classes=65) -> None:
        super().__init__()
        print('Separate Model')
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        # self.transformer1 = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.transformer1 = MAE_Encoder(224,16,192,8,3,0.5).transformer
        self.transformer2 = MAE_Encoder(224,16,192,8,3,0.5).transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1] * 2, num_classes)

    def forward(self, img1, img2):
        patches1 = self.patchify(img1)
        patches2 = self.patchify(img2)

        patches1 = rearrange(patches1, 'b c h w -> (h w) b c')
        patches2 = rearrange(patches2, 'b c h w -> (h w) b c')

        patches1 = patches1 + self.pos_embedding
        patches2 = patches2 + self.pos_embedding

        patches1 = torch.cat([self.cls_token.expand(-1, patches1.shape[1], -1), patches1], dim=0)
        patches2 = torch.cat([self.cls_token.expand(-1, patches2.shape[1], -1), patches2], dim=0)

        patches1 = rearrange(patches1, 't b c -> b t c')
        features1 = self.layer_norm(self.transformer1(patches1))
        features1 = rearrange(features1, 'b t c -> t b c')

        patches2 = rearrange(patches2, 't b c -> b t c')
        features2 = self.layer_norm(self.transformer2(patches2))
        features2 = rearrange(features2, 'b t c -> t b c')

        embed = torch.cat([features1[0], features2[0]], dim=1)

        logits = self.head(embed)
        return logits

class TS_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=200) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img1, img2):
        patches1 = self.patchify(img1)
        patches2 = self.patchify(img2)

        patches1 = rearrange(patches1, 'b c h w -> (h w) b c')
        patches2 = rearrange(patches2, 'b c h w -> (h w) b c')

        patches1 = patches1 + self.pos_embedding
        patches2 = patches2 + self.pos_embedding

        patches = torch.cat([patches1, patches2], dim=0)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


class ViT_OneDrug_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=65) -> None:
        super().__init__()
        print('Unifying Model')
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.shuffle = encoder.shuffle
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1] *2 , num_classes)


    def forward_once(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        cls_feature = features[:, 0, :]                           
        return cls_feature
    
    def forward(self,img1,img2):
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)
        combined = torch.cat([emb1, emb2], dim=1)
        logits = self.head(combined)
        return logits