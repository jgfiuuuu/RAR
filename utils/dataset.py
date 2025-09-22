import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,RandFlipd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224],is_labeled=True):

        super(QaTa, self).__init__()

        self.mode = mode
        self.is_labeled = is_labeled

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        if mode == 'pretrain':
            self.image_list = self.image_list[:int(0.25*len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.25*len(self.caption_list))]

        if mode == 'semi':
            self.labeled_image_list = self.image_list[:int(0.25 * len(self.image_list))]
            self.unlabeled_image_list = self.image_list[int(0.25 * len(self.image_list)):]

            self.labeled_caption_list = self.caption_list[:int(0.25 * len(self.caption_list))]
            self.unlabeled_caption_list = self.caption_list[int(0.25 * len(self.caption_list)):]

            self.image_list = self.labeled_image_list + self.unlabeled_image_list
            self.caption_list = self.labeled_caption_list + self.unlabeled_caption_list
        elif mode == 'valid':
            pass
        else:
            pass   # for mode is 'test'

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):


        trans = self.transform(self.image_size)
        
        
        image_ids = os.path.join(self.root_path,'Images',self.image_list[idx].replace('mask_',''))
        gt = os.path.join(self.root_path,'GTs', self.image_list[idx])
        
        
        
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image_ids, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']
        unique_vals = torch.unique(gt).cpu().tolist()
        allowed = [0, 1, 255, False, True]
        if any(val not in allowed for val in unique_vals):
            print("Warning: non-binary values found in gt! Unique values:", unique_vals)
        if gt.dtype == torch.bool:
            gt = gt.int()
        elif gt.max() > 1:
            gt = torch.where(gt > 0, 1, 0)
            #gt = torch.where(gt==255,1,0)
        else:
            gt = gt.int()
        
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 

        
        if image.shape[0] == 1:   
            image = repeat(image,'1 h w -> c h w',c=3)
            #gt = repeat(gt,'1 h w -> c h w',c=3)
        
        if self.mode == 'semi' and idx >= len(self.labeled_image_list):
            flag = 0
            placeholder_gt = torch.zeros_like(gt, dtype=torch.int) #covid  breast_tumors
            #placeholder_gt = torch.zeros(3, *self.image_size, dtype=torch.int) #mosmed
            #print(image.shape,placeholder_gt.shape,gt.shape)
            
        
            return ([image, text, placeholder_gt], placeholder_gt,flag,image_ids)     
        else:   
            flag = 1
            return ([image, text, gt], gt,flag,image_ids)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])
        return trans