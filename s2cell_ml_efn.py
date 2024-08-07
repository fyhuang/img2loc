"""Multi-label (one for each parent S2 cell) version of classifier,
based on EfficientNet-B5.
"""

import argparse
import math
import multiprocessing
import heapq
import dataclasses
import collections

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

from mlutil import label_mapping, s2cell_mapping, geoguessr_score
from datasets import World1, Img2LocCombined


FT_SCHEDULE_PATH = "finetune_s2cell_efn.yaml"


# Define a LightningModule for the classifier
class S2CellClassifierEfn(L.LightningModule):
    def __init__(self, label_mapping, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate

        self.label_mapping = label_mapping
        self.s2cell_mapping = s2cell_mapping.S2CellMapping.from_label_mapping(label_mapping)
        num_classes = len(label_mapping)

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
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(hidden_size, num_classes), # out is 1776
            nn.Sigmoid(),
        )

        # Example input array (for logging graph)
        self.example_input_array = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

        self.subset_accuracy = torchmetrics.classification.MultilabelExactMatch(num_labels=num_classes)
        self.f1_score = torchmetrics.classification.MultilabelF1Score(num_labels=num_classes)
        self.geo_score = geoguessr_score.GeoguessrScore()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def logits_to_latlng(self, z):
        # Pick the best S2 cell from model's output and convert to lat/lng

        assert z.shape[1] == len(self.label_mapping)

        # Convert all output labels
        preds = z > 0.5
        pred_ll = torch.zeros((z.shape[0], 2), dtype=torch.float32, device=z.device)

        for row_num in range(z.shape[0]):
            tokens = [self.label_mapping.get_name(i) for i in range(len(self.label_mapping)) if preds[row_num, i]]
            if len(tokens) == 0:
                # No prediction
                continue
            best_cell_id = self.s2cell_mapping.token_list_to_prediction(tokens)
            lat_lng = best_cell_id.to_lat_lng()
            pred_ll[row_num, 0] = lat_lng.lat().degrees
            pred_ll[row_num, 1] = lat_lng.lng().degrees

        return pred_ll


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
        self.log(f"{log_name_prefix}_f1{log_name_suffix}", self.f1_score, on_step=log_step, on_epoch=(not log_step))

        # Compute geoguessr score from target lat/lng
        if log_name_prefix == "val":
            target_ll = targets[:, :2]
            pred_ll = self.logits_to_latlng(z)
            self.geo_score(pred_ll, target_ll)
            self.log(f"{log_name_prefix}_geo{log_name_suffix}", self.geo_score, on_step=log_step, on_epoch=(not log_step))

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
            lr=self.learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=10,
                ),
                "monitor": "train_acc_step",
                "interval": "step",
                "frequency": 1000,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu")
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
    combined_dataset = Img2LocCombined()
    if args.mode != "overfit":
        train_dataloader = combined_dataset.train_dataloader()
        val_dataloader = combined_dataset.val_dataloader()
    else:
        if args.overfit == "1":
            train_dataloader = world1.overfit_dataloader_one()
            val_dataloader = world1.overfit_dataloader_one(val=True)
        elif args.overfit == "5":
            train_dataloader = world1.overfit_dataloader_five()
            val_dataloader = world1.overfit_dataloader_five(val=True)
        else:
            raise NotImplementedError()

    efn_model = S2CellClassifierEfn(
        label_mapping=world1.label_mapping,
        #learning_rate=1.5e-3,
        learning_rate=5.0e-4,
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
    elif args.mode == "overfit":
        check_val_every_n_epoch = 3

    val_acc_patience = 5 // check_val_every_n_epoch
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
        limit_val_batches=args.limit_batches // 20,
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
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
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