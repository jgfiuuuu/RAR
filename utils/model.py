import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import math
from einops import rearrange, repeat
from transformers import AutoModel
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
import numpy as np
import matplotlib.pyplot as plt

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, 2, activation)
    def forward(self, x, skip_x):
        out = self.up(x)      
        x = torch.cat([out, skip_x], dim=1) 
        return self.nConvs(x)
class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)


        return {'feature':output['hidden_states']}
    
# Vision Encoder
class VisionModel(nn.Module):#

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.spatial_dim = 768

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)


        return output['hidden_states']

    
class Expertfg(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expertfg, self).__init__()
        
        self.up4 = UpBlock(768, 384, nb_Conv=2)
        self.up3 = UpBlock(384, 192, nb_Conv=2)
        self.up2 = UpBlock(192, 96, nb_Conv=2)       

    def forward(self, x):
       

        x3 = self.up4(x[-1], x[-2])        
        x2 = self.up3(x3, x[-3])
        x1 = self.up2(x2, x[-4])
       
        
        
        return [x3,x2,x1]

class RARSeg(nn.Module):
    def __init__(self, bert_type, vision_type, project_dim=768,mode='pretrain'):
        super(RARSeg, self).__init__()
        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        self.expert_fg = Expertfg(input_dim=768, hidden_dim=128, output_dim=1)  # Foreground Expert
        self.up1 = UpBlock(1, 1, nb_Conv=2)
        self.decoder1 = SubpixelUpsample(2,96,24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        

    def forward(self, data):
        # Encoder
        image, text, gt = data

        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)
        image_features = self.encoder(image)
        text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_embeds = text_output['feature'][-1]
        image_features = [feat for feat in image_features]
        
        has_text = (text['attention_mask'].sum(dim=1) > 2)
        b = image_features[-1].shape[0]

        
        fg_output_img = self.expert_fg(image_features)    
        fg_output_preds = self.decoder1(fg_output_img[-1])
        fg_output = self.out(fg_output_preds).sigmoid()  
            
        
        img = torch.mean(image_features[-1], dim=(2,3), keepdim=False)
        img = img / img.norm(p=2)
        txt = torch.mean(text_embeds, dim=1, keepdim=False)
        txt = txt / txt.norm(p=2)
        
        if text_embeds is not None:
            txt_mask_expanded = has_text.view(b, 1).expand_as(txt)
            txt = torch.where(txt_mask_expanded, txt, torch.zeros_like(txt))
        
        return fg_output,img,txt
    
    
