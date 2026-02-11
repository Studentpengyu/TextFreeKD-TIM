import torch
from torch.utils.data import DataLoader
from utils.dataset import QaTa
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper import LanGuideMedSegWrapper

import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

    args = get_parser()
    print("cuda:",torch.cuda.is_available())

    ds_train = QaTa(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='train')

    ds_valid = QaTa(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')


    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

    model = LanGuideMedSegWrapper(args)
    checkpoint = torch.load('./save_model/medseg.ckpt', map_location='cpu')["state_dict"] # select medseg and medseg-mos2 for COV19 and Mos
    new_checkpoint = {}

    for key, value in checkpoint.items():
        # 去除 'model.' 前缀
        if key.startswith('model.'):
            new_key = key[len('model.'):]
        else:
            new_key = key
        new_checkpoint[new_key] = value

    model.model_teacher.load_state_dict(new_checkpoint,strict=True)

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='student_val_dice',
        save_top_k=1,
        mode='max',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'student_val_dice',
                            patience=args.patience,
                            mode = 'max'
    )

    ## 2. setting trainer

    trainer = pl.Trainer(logger=True,
                        min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                        accelerator='gpu', 
                        devices=args.device,
                        callbacks=[model_ckpt, early_stopping],
                        enable_progress_bar=False,
                        ) 

    ## 3. start training
    print('start training')
    trainer.fit(model,dl_train,dl_valid)
    print('done training')

