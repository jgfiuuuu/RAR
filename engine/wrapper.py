from utils.model import RARSeg
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import os
import cv2
import sys
import numpy as np
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from thop import profile, clever_format

def cosine_similarity(a, b):

    a = F.normalize(a, dim=-1)

    b = F.normalize(b, dim=-1)

    return torch.matmul(a, b.T) 

def top_k(sim_matrix, k):
    topk_weights, topk_indices = torch.topk(sim_matrix, k=k, dim=-1)
    return topk_weights, topk_indices

def contrastive_loss(feat_q, feat_pos, feat_neg_all, temperature=0.1):
    feat_q = F.normalize(feat_q, dim=-1)
    feat_pos = F.normalize(feat_pos, dim=-1)
    feat_neg_all = F.normalize(feat_neg_all, dim=-1)
    logits_pos = (feat_q * feat_pos).sum(dim=-1, keepdim=True) / temperature
    logits_neg = torch.matmul(feat_q, feat_neg_all.T) / temperature
    logits = torch.cat([logits_pos, logits_neg], dim=1)
    labels = torch.zeros(feat_q.size(0), dtype=torch.long).to(feat_q.device)
    return F.cross_entropy(logits, labels)

class RARSegWrapper(pl.LightningModule):

    def __init__(self, args,mode = 'pretrain',weight_consistency=1, weight_contrastive=0.05):
        
        super(RARSegWrapper, self).__init__()
        self.mode = mode
        self.weight_consistency = weight_consistency
        self.weight_contrastive = weight_contrastive
        self.model = RARSeg(args.bert_type, args.vision_type, args.project_dim,mode)
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False    
        self.ema_decay = 0.99  

        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        #key: image_id, flag,value: {'img_feat': tensor, 'txt_feat': tensor}
        self.memory_bank = {}
    
    
        self.memory_bank_ema_decay = 0.99
        
        #self.save_hyperparameters()
        self.save_hyperparameters('weight_consistency', 'weight_contrastive')


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       
       return self.model.forward(x)
    
    def update_ema_variables(self, model, ema_model, global_step,alpha=0.99):

        alpha = min(1 - 1 / (global_step + 1), alpha)  
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
    def _get_consistency_weight(self):

        rampup_length = 10
        end_weight = self.weight_consistency
        if rampup_length == 0:
            return end_weight
        current_epoch = float(self.trainer.current_epoch)
        if current_epoch > rampup_length:
            weight = end_weight
        else:
            phase = 1.0 - current_epoch / rampup_length
            weight = end_weight * np.exp(-5.0 * phase * phase)

        return weight

    def _compute_retrieval_losses(self, query_img_feats, query_txt_feats, k=5, temperature=0.1):

     
        if len(self.memory_bank) < k: 
            return None
            
        db_ids = list(self.memory_bank.keys())
        db_img_features = torch.stack([self.memory_bank[id]['img_feat'] for id in db_ids]).to(self.device)
        db_text_features = torch.stack([self.memory_bank[id]['txt_feat'] for id in db_ids]).to(self.device)
        effective_k = min(k, db_text_features.shape[0])

        sim_matrix_text = cosine_similarity(query_txt_feats, db_text_features)
        topk_text_weights, topk_text_indices = top_k(sim_matrix_text, k=effective_k)
        retrieved_img_features_by_text = db_img_features[topk_text_indices]
        softmax_weights_text = F.softmax(topk_text_weights, dim=-1)
        positive_img_features_by_text = (softmax_weights_text.unsqueeze(-1) * retrieved_img_features_by_text).sum(dim=1)
        
        loss_cross_modal = contrastive_loss(
            feat_q=query_img_feats,
            feat_pos=positive_img_features_by_text,
            feat_neg_all=db_img_features,
            temperature=temperature
        )
        

        sim_matrix_image = cosine_similarity(query_img_feats, db_img_features)
        topk_img_weights, topk_img_indices = top_k(sim_matrix_image, k=effective_k)
        

        retrieved_img_features_by_img = db_img_features[topk_img_indices]
        softmax_weights_img = F.softmax(topk_img_weights, dim=-1)
        positive_img_features_by_img = (softmax_weights_img.unsqueeze(-1) * retrieved_img_features_by_img).sum(dim=1)

        loss_image_image = contrastive_loss(
            feat_q=query_img_feats,
            feat_pos=positive_img_features_by_img,
            feat_neg_all=db_img_features,
            temperature=temperature
        )


        return {
            'cross_modal': loss_cross_modal,
            'image_image': loss_image_image
        }


    def shared_step(self, batch, batch_idx,exists_zero_image = False):
        x, y ,flag,image_ids= batch
        if torch.all(y == 0):
            print("y is all zeros")

        if not self.trainer.training:     
            preds,img_features_list,txts = self(x)
            loss = self.loss_fn(preds, y)


            return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()} 




        preds, img_features_list, txt_features_list = self(x)     
        with torch.no_grad():
            ema_preds, _, _ = self.ema_model(x)  

        current_img_feats = img_features_list
        current_txt_feats = txt_features_list

        labeled_mask = (flag == 1)
        unlabeled_mask = (flag == 0)

        if labeled_mask.any():
            labeled_ids = [image_ids[i] for i, is_labeled in enumerate(labeled_mask) if is_labeled]
            labeled_img_feats = current_img_feats[labeled_mask]
            labeled_txt_feats = current_txt_feats[labeled_mask]

            for i, sample_id in enumerate(labeled_ids):
                if sample_id in self.memory_bank:
                    old_entry = self.memory_bank[sample_id]
                    new_img_feat = self.memory_bank_ema_decay * old_entry['img_feat'] + \
                                (1 - self.memory_bank_ema_decay) * labeled_img_feats[i].detach().cpu()
                    new_txt_feat = self.memory_bank_ema_decay * old_entry['txt_feat'] + \
                                (1 - self.memory_bank_ema_decay) * labeled_txt_feats[i].detach().cpu()
                    self.memory_bank[sample_id] = {'img_feat': new_img_feat, 'txt_feat': new_txt_feat}
                else:
                    self.memory_bank[sample_id] = {
                        'img_feat': labeled_img_feats[i].detach().cpu(),
                        'txt_feat': labeled_txt_feats[i].detach().cpu()
                    }
            
        normal_loss_sum = 0.0
        placeholder_loss_sum = 0.0
        loss_ctr_total = 0.0

        if labeled_mask.any():
            loss_sup = self.loss_fn(preds[labeled_mask], y[labeled_mask])
            normal_loss_sum += loss_sup
        if unlabeled_mask.any():
            y_pseudo = (ema_preds[unlabeled_mask] > 0.5).float() 
            loss_consistency = self.loss_fn(preds[unlabeled_mask], y_pseudo)
            placeholder_loss_sum += loss_consistency

        if self.mode == 'semi':
            
            query_img_unlabeled = current_img_feats[unlabeled_mask]
            query_txt_unlabeled = current_txt_feats[unlabeled_mask]
            
            retrieval_losses = self._compute_retrieval_losses(
                query_img_feats=query_img_unlabeled,
                query_txt_feats=query_txt_unlabeled,
                k=5,
                temperature=0.07
            )
            loss_ctr_cross = retrieval_losses['cross_modal']
            loss_ctr_img = retrieval_losses['image_image']
            
            
            loss_ctr_total = loss_ctr_cross + loss_ctr_img

        processed_y = y.clone().detach()
        if unlabeled_mask.any():
            pseudo_labels = (ema_preds[unlabeled_mask] > 0.5)
            processed_y[unlabeled_mask] = pseudo_labels.to(processed_y.dtype)
        self.update_ema_variables(self.model, self.ema_model, self.trainer.global_step, self.ema_decay)
        loss = normal_loss_sum + self.weight_consistency * placeholder_loss_sum + self.weight_contrastive * loss_ctr_total
        
        return {'loss': loss, 'preds': preds.detach(), 'y': processed_y.detach()}
        
            
        
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
        
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    

    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)

    def on_save_checkpoint(self, checkpoint):
        print("\nSaving memory bank to checkpoint...")
        checkpoint['memory_bank'] = self.memory_bank
        print(f"Memory bank with {len(self.memory_bank)} items saved.")

    def on_load_checkpoint(self, checkpoint):
        self.memory_bank = checkpoint.get('memory_bank', {})
        print("\nLoading memory bank from checkpoint...")
        print(f"Memory bank with {len(self.memory_bank)} items loaded.")