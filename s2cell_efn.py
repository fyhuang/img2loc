import argparse
import math
import multiprocessing
import heapq
import dataclasses

import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
import numpy as np
import pandas
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping

import torchmetrics
import webdataset as wds
import s2sphere
import tqdm

import label_mapping
from datasets import Im2gps2007


FT_SCHEDULE_PATH = "finetune_s2cell_efn.yaml"


# Define a LightningModule for the classifier
class S2CellClassifierEfn(L.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=[])

        efn = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)

        self.features = efn.features
        self.avgpool = efn.avgpool

        lastconv_output_channels = efn.classifier[1].in_features # 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes), # out is 1776
        )

        torch.nn.init.xavier_uniform_(self.classifier[1].weight)

        self.accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = nn.CrossEntropyLoss()(z, y)
        self.log("train_loss", loss, prog_bar=True)

        preds = torch.argmax(z, dim=1)
        self.accuracy(preds, y)
        self.log('train_acc_step', self.accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        val_loss = nn.CrossEntropyLoss()(z, y)
        self.log("val_loss", val_loss, prog_bar=True)

        preds = torch.argmax(z, dim=1)
        self.accuracy(preds, y)
        self.log('val_acc', self.accuracy, on_step=False, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        test_loss = nn.CrossEntropyLoss()(z, y)
        self.log("test_loss", test_loss, prog_bar=True)

        preds = torch.argmax(z, dim=1)
        self.accuracy(preds, y)
        self.log('test_acc', self.accuracy, on_step=False, on_epoch=True)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3),
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--limit_batches", type=float, default=1.0)

    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--gen_ft_sched_only", action="store_true")

    args = parser.parse_args()

    im2gps2007 = Im2gps2007()
    train_dataloader = im2gps2007.train_dataloader()
    val_dataloader = im2gps2007.val_dataloader()

    # Full training (start with frozen layers, then unfreeze)

    # Temporarily load checkpoint manually (since we changed the model interface)
    #checkpoint = torch.load("checkpoints/s2cell_predict/version2.ckpt", **LOAD_CHECKPOINT_MAP_LOCATION)
    mnet3_model = S2CellClassifierEfn(
        num_classes=len(im2gps2007.mapping),
        learning_rate=1e-3,
    )
    #LOAD_CHECKPOINT_MAP_LOCATION = {}
    #if not torch.cuda.is_available():
    #    LOAD_CHECKPOINT_MAP_LOCATION = {"map_location": torch.device("cpu")}

    #mnet3_model.load_state_dict(checkpoint["state_dict"])
    #mnet3_model = S2CellClassifierMnet3.load_from_checkpoint("checkpoints/s2cell_predict/version1.ckpt", num_classes=len(mapping))

    callbacks = []
    if not args.fast_dev_run:
        callbacks.extend([
            FinetuningScheduler(
                gen_ft_sched_only=args.gen_ft_sched_only,
                ft_schedule=FT_SCHEDULE_PATH,
            ),
            FTSCheckpoint(
                monitor="val_acc",
                mode="max",
                save_last=True,
                save_top_k=5,
            ),
            FTSEarlyStopping(
                patience=5,
                monitor="val_acc",
                mode="max",
                verbose=True,
            ),
        ])
    callbacks.append(
        L.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step"
        ),
    )

    fast_dev_run_args = {}
    if args.fast_dev_run:
        # FinetuningScheduler seems to fail if checkpointing/logging not enabled, so only keep the
        # flags that limit runtime
        fast_dev_run_args = {
            "max_epochs": 1,
            "max_steps": 1,
            "num_sanity_val_steps": 0,
            "val_check_interval": 1.0,
            "check_val_every_n_epoch": 1,
        }

    trainer = L.Trainer(
        accelerator=args.accelerator,
        callbacks=callbacks,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        **fast_dev_run_args,
    )
    trainer.fit(
        model=mnet3_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.ckpt_path,
    )
    print("Training done")


if __name__ == "__main__":
    main()