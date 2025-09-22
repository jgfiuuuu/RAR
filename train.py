import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import QaTa
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper import RARSegWrapper

import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg


if __name__ == '__main__':
    set_seed(0)
    args = get_parser()
    print("cuda:", torch.cuda.is_available())

    print("Preparing labeled data...")
    ds_labeled = QaTa(csv_path=args.train_csv_path,
                      root_path=args.train_root_path,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='pretrain')
    dl_labeled = DataLoader(ds_labeled, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)

    ds_valid = QaTa(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)

    print("Start pre-training with labeled data...")
    model = RARSegWrapper(args,mode ='pretrain')


    model_ckpt_pre = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )
    early_stopping_pre = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )

    trainer_pre = pl.Trainer(
        logger=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.device,
        callbacks=[model_ckpt_pre, early_stopping_pre],
        enable_progress_bar=False,
    )
    trainer_pre.fit(model, dl_labeled, dl_valid)
    print("Pre-training complete.")


    print("Preparing for semi-supervised training...")
    model = RARSegWrapper(args,mode ='semi')


    CKPT_PATH = './save_model/medseg.ckpt'


    checkpoint = torch.load(CKPT_PATH, map_location='cpu')


    print("Restoring memory_bank...")
    model.on_load_checkpoint(checkpoint)


    print("Loading model weights...")
    model.load_state_dict(checkpoint['state_dict'])

    ds_semi = QaTa(csv_path=args.train_csv_path,
                   root_path=args.train_root_path,
                   tokenizer=args.bert_type,
                   image_size=args.image_size,
                   mode='semi')

    dl_semi = DataLoader(ds_semi, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)


    model_ckpt_semi = ModelCheckpoint(
        dirpath='./semi_supervised',
        filename='semi_supervised',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )
    early_stopping_semi = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )

    trainer_semi = pl.Trainer(
        logger=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.device,
        callbacks=[model_ckpt_semi, early_stopping_semi],
        enable_progress_bar=False,
    )
    print("Start semi-supervised training...")
    trainer_semi.fit(model, dl_semi, dl_valid)
    print("Semi-supervised training complete.")


    print("Testing the semi-supervised model...")
    model = RARSegWrapper(args)
    checkpoint = torch.load('./semi_supervised/semi_supervised.ckpt', map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    ds_test = QaTa(csv_path=args.test_csv_path,
                   root_path=args.test_root_path,
                   tokenizer=args.bert_type,
                   image_size=args.image_size,
                   mode='test')
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    trainer_test = pl.Trainer(accelerator='gpu', devices=args.device)
    model.eval()
    trainer_test.test(model, dl_test)
