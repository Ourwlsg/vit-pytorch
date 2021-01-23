from __future__ import print_function

import glob
import shutil
from itertools import chain
import os
import random
import zipfile

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from lib.CassavaDataset import CassavaDataset
from vit_pytorch.efficient import ViT
from cfg.cfg import cfg

from lib.utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_category_list,
)
from lib.loss import *

print(f"Torch: {torch.__version__}")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    DIR_CV = './BBN/cassava/data/new_cv20/'
    # DIR_CV = '/home/zhucc/kaggle/pytorch_classification/data/cv/'
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    RandomAugment = True
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID

    logger, log_file = create_logger(cfg)
    for k in cfg.K_FOLD:
        train_label_file = DIR_CV + 'fold_' + str(k) + '/train.txt'
        val_label_file = DIR_CV + 'fold_' + str(k) + '/val.txt'
        train_set = CassavaDataset(train_label_file, mode="train", transform_name="RandomAugment")
        valid_set = CassavaDataset(val_label_file, mode="valid", transform_name=None)

        trainLoader = DataLoader(
            train_set,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=True
        )

        validLoader = DataLoader(
            valid_set,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )
        annotations = train_set.get_annotations()
        num_classes = train_set.get_num_classes()

        num_class_list, cat_list = get_category_list(annotations, num_classes)
        para_dict = {
            "num_classes": num_classes,
            "num_class_list": num_class_list,
            "cfg": cfg,
            "device": device,
        }

        efficient_transformer = Linformer(
            dim=1024,
            seq_len=64 + 1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )

        model = ViT(
            dim=1024,
            image_size=cfg.INPUT_SIZE,
            patch_size=64,
            num_classes=cfg.CLS_NUM,
            transformer=efficient_transformer,
            channels=cfg.INPUT_CHANNEL,
        ).to(device)

        criterion = {
            'CrossEntropy': lambda: CrossEntropy(para_dict=para_dict),
            'CSCE': lambda: CSCE(para_dict=para_dict),
            'LDAMLoss': lambda: LDAMLoss(para_dict=para_dict),
            'SymmetricCrossEntropy': lambda: SymmetricCrossEntropy(alpha=0.1, beta=1.0, num_classes=5),
            # 'bi_tempered_logistic_loss': lambda: bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5),
            'LabelSmoothingCrossEntropy': lambda: LabelSmoothingCrossEntropy(),
        }[cfg.LOSS_TYPE]()
        # optimizer
        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer)

        model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, f"fold_{k}", "models")
        code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, f"fold_{k}", "codes")
        tensorboard_dir = (
            os.path.join(cfg.OUTPUT_DIR, cfg.NAME, f"fold_{k}", "tensorboard")
            if cfg.TENSORBOARD_ENABLE
            else None
        )
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        else:
            logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            if not click.confirm(
                    "\033[1;31;40mContinue and override the former directory?\033[0m",
                    default=False,
            ):
                exit(0)
            shutil.rmtree(code_dir)
            if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
        print("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*", "*data*"
        )
        # shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)
        if tensorboard_dir is not None:
            dummy_input = torch.rand((1, 3) + (cfg.INPUT_SIZE, cfg.INPUT_SIZE)).to(device)
            writer = SummaryWriter(log_dir=tensorboard_dir)
            # writer.add_graph(model if cfg.CPU_MODE else model, (dummy_input,))
        else:
            writer = None

        best_result, best_epoch, start_epoch = 0, 0, 1
        logger.info("-------------------Train start :{}-------------------".format(cfg.BACKBONE))
        for epoch in range(start_epoch, cfg.MAX_EPOCH+1):

            epoch_loss = 0
            epoch_accuracy = 0

            for data, label in tqdm(trainLoader):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                print(f"acc = {acc}")
                epoch_accuracy += acc / len(trainLoader)
                epoch_loss += loss / len(trainLoader)
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            writer.add_scalar('learning_rate', lr, epoch)
            loss_dict, acc_dict = {"train_loss": epoch_loss}, {"train_acc": epoch_accuracy}
            labels_epoch = []
            prediction_epoch = []
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                func = torch.nn.Softmax(dim=1)
                for data, label in validLoader:
                    # labels_epoch.extend(label.numpy())
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()

                    # score_result = func(val_output)
                    # now_result = torch.argmax(score_result, 1)
                    # prediction_epoch.extend(now_result.cpu().numpy())

                    epoch_val_accuracy += acc / len(validLoader)
                    epoch_val_loss += val_loss / len(validLoader)
                loss_dict["valid_loss"], acc_dict["valid_acc"] = epoch_val_loss, epoch_val_accuracy
                # result_epoch = classification_report(labels_epoch, prediction_epoch, labels=[0, 1, 2, 3, 4],
                #                                      target_names=target_names,
                #                                      output_dict=True, digits=3)
                # print(classification_report(labels_epoch, prediction_epoch, labels=[0, 1, 2, 3, 4],
                #                             target_names=target_names,
                #                             digits=3))

            print(
                f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
            # save checkpoint

            model_save_path = os.path.join(model_dir, "epoch_{}_{}.pth".format(epoch, epoch_val_accuracy))
            if epoch % cfg.SAVE_STEP == 0:
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_save_path)

            if cfg.TENSORBOARD_ENABLE:
                writer.add_scalars("scalar/acc", acc_dict, epoch)
                writer.add_scalars("scalar/loss", loss_dict, epoch)
                # writer.add_scalar('ACC/val', result_epoch['accuracy'], epoch)
                # for metric in ['precision', 'recall', 'f1-score']:
                #     writer.add_scalars(f'Metrics/{metric}',
                #                        {target_name: result_epoch[target_name][metric] for target_name in
                #                         target_names},
                #                        epoch)
        if cfg.TENSORBOARD_ENABLE:
            writer.close()
        logger.info(
            "-------------------Train Finished :{}-------------------".format(cfg.NAME)
        )