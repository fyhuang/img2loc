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
import lightning.pytorch.tuner
from lightning.pytorch.loggers import TensorBoardLogger
from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping

import torchmetrics
import webdataset as wds
import s2sphere
import tqdm

from mlutil import label_mapping
from datasets import Im2gps2007


FT_SCHEDULE_PATH = "finetune_s2cell_efn.yaml"


# Define a LightningModule for the classifier
class S2CellClassifierEfn(L.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=[])

        efn = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)

        # TODO: Change last layer to output more features (3072)
        #features_layers = []
        #for i, m in enumerate(efn.features):
        #    if i != len(efn.features) - 1:
        #        features_layers.append(m)
        #lastconv_input_channels = efn.features[-1].in_channels
        self.features = efn.features
        #self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # in_features = 2048
        avgpool_out_features = \
            efn.classifier[1].in_features * \
                self.avgpool.output_size
        hidden_size = 2048
        self.classifier = nn.Sequential(
            nn.Linear(avgpool_out_features, hidden_size),
            nn.SiLU(inplace=True),
            #nn.Dropout(p=0.4, inplace=True),
            nn.Linear(hidden_size, num_classes), # out is 1776
        )

        # Example input array (for logging graph)
        self.example_input_array = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

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
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.99, # disable for this phase
                    patience=40
                ),
                "monitor": "val_acc",
                "interval": "epoch",
                # TODO: this needs to be the same as check_val_every_n_epoch
                "frequency": 3,
            },
            #"lr_scheduler": {
            #    "scheduler": optim.lr_scheduler.LinearLR(
            #        optimizer,
            #        start_factor=0.7,
            #        end_factor=1.0,
            #        total_iters=40,
            #    ),
            #},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--limit_batches", type=float, default=None)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "fast_dev_run", "gen_ft_sched", "lr_find", "overfit"],
        default="fast_dev_run",
    )
    parser.add_argument("--overfit", choices=["1", "5"], default="1")

    args = parser.parse_args()

    im2gps2007 = Im2gps2007()
    if args.mode != "overfit":
        train_dataloader = im2gps2007.train_dataloader()
        val_dataloader = im2gps2007.val_dataloader()
    else:
        if args.overfit == "1":
            train_dataloader = im2gps2007.overfit_dataloader_one()
            val_dataloader = im2gps2007.overfit_dataloader_one(val=True)
        elif args.overfit == "5":
            train_dataloader = im2gps2007.overfit_dataloader_five()
            val_dataloader = im2gps2007.overfit_dataloader_five(val=True)
        else:
            raise NotImplementedError()

    efn_model = S2CellClassifierEfn(
        num_classes=len(im2gps2007.mapping),
        learning_rate=1.5e-3,
    )

    fast_dev_run_args = {}
    check_val_every_n_epoch = 1
    if args.mode == "fast_dev_run":
        # FinetuningScheduler seems to fail if checkpointing/logging not enabled, so only keep the
        # flags that limit runtime
        fast_dev_run_args = {
            "max_epochs": 1,
            "max_steps": 1,
            "num_sanity_val_steps": 0,
            "val_check_interval": 1.0,
        }
        check_val_every_n_epoch = 2
    elif args.mode == "overfit":
        check_val_every_n_epoch = 3

    val_acc_patience = 60 // check_val_every_n_epoch
    callbacks = []
    if args.mode in ["train", "overfit", "gen_ft_sched"]:
        callbacks.extend([
            FinetuningScheduler(
                gen_ft_sched_only=(args.mode == "gen_ft_sched"),
                ft_schedule=FT_SCHEDULE_PATH,
            ),
            FTSCheckpoint(
                monitor="val_acc",
                mode="max",
                save_last=True,
                save_top_k=3,
            ),
            FTSEarlyStopping(
                patience=val_acc_patience,
                monitor="val_acc",
                mode="max",
                verbose=True,
            ),
        ])

    callbacks.extend([
        L.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step"
        ),
    ])

    trainer = L.Trainer(
        accelerator=args.accelerator,
        callbacks=callbacks,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=L.pytorch.loggers.TensorBoardLogger(
            save_dir=".",
            log_graph=True,
        ),
        **fast_dev_run_args,
    )

    if args.mode == "lr_find":
        tuner = lightning.pytorch.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(
            model=efn_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        print(lr_finder.results)
    else:
        trainer.fit(
            model=efn_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.ckpt_path,
        )
        print("Training done")


if __name__ == "__main__":
    main()