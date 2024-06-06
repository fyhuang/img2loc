"""Multi-label (one for each parent S2 cell) version of classifier,
based on EfficientNet-B5.
"""

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
from datasets import World1


FT_SCHEDULE_PATH = "finetune_s2cell_efn.yaml"


# Define a LightningModule for the classifier
class S2CellClassifierEfn(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        efn = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)

        self.features = efn.features
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
            nn.Sigmoid(),
        )

        # Example input array (for logging graph)
        self.example_input_array = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

        self.subset_accuracy = torchmetrics.classification.MultilabelExactMatch(num_labels=num_classes)
        self.f1_score = torchmetrics.classification.MultilabelF1Score(num_labels=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _infer_with_loss_acc(self, x, targets, log_name_prefix, log_step=True):
        # extract ML labels from targets
        y = targets[:, 3:]

        log_name_suffix = ""
        if log_step:
            log_name_suffix = "_step"

        z = self.forward(x)
        loss = nn.BCELoss()(z, y)
        self.log(f"{log_name_prefix}_loss{log_name_suffix}", loss, prog_bar=True, on_step=log_step, on_epoch=(not log_step))

        self.subset_accuracy(z, y)
        self.log(f"{log_name_prefix}_acc{log_name_suffix}", self.subset_accuracy, prog_bar=True, on_step=log_step, on_epoch=(not log_step))

        self.f1_score(z, y)
        self.log(f"{log_name_prefix}_f1{log_name_suffix}", self.f1_score, prog_bar=True, on_step=log_step, on_epoch=(not log_step))

        return loss, z

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self._infer_with_loss_acc(x, y, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self._infer_with_loss_acc(x, y, "val", log_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=1.5e-3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.33,
                    patience=10,
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

    world1 = World1()
    if args.mode != "overfit":
        train_dataloader = im2gps2007.train_dataloader()
        val_dataloader = im2gps2007.val_dataloader()
    else:
        if args.overfit == "1":
            train_dataloader = world1.overfit_dataloader_one()
            val_dataloader = world1.overfit_dataloader_one(val=True)
        else:
            raise NotImplementedError()

    efn_model = S2CellClassifierEfn(
        num_classes=len(world1.label_mapping),
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
        #callbacks.extend([
        #    FinetuningScheduler(
        #        gen_ft_sched_only=(args.mode == "gen_ft_sched"),
        #        ft_schedule=FT_SCHEDULE_PATH,
        #    ),
        #    FTSCheckpoint(
        #        monitor="val_acc",
        #        mode="max",
        #        save_last=True,
        #        save_top_k=3,
        #    ),
        #    FTSEarlyStopping(
        #        patience=val_acc_patience,
        #        monitor="val_acc",
        #        mode="max",
        #        verbose=True,
        #    ),
        #])
        callbacks.extend([
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_last=True,
                save_top_k=3,
            ),
            L.pytorch.callbacks.EarlyStopping(
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