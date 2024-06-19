"""Multi-label (one for each parent S2 cell) classifier system.

Using pluggable models.
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
from datasets import World1, Img2LocCombined, Im2gps2007


def make_efn_model():
    efn = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)

    # in_features = 2048
    avgpool_out_features = \
        efn.classifier[1].in_features * \
            self.avgpool.output_size
    hidden_size = 2048
    efn.classifier = nn.Sequential(
        nn.Linear(avgpool_out_features, hidden_size),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(hidden_size, num_classes), # out is 1776
        nn.Sigmoid(),
    )

    return efn


def efn_linear_init(layer):
    init_range = 1.0 / math.sqrt(layer.out_features)
    nn.init.uniform_(layer.weight, -init_range, init_range)
    nn.init.zeros_(layer.bias)

def make_efnv2_s_model(num_classes, dropout, hidden):
    efn = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # dropout = 0.2
    # norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
    # last_channel = 1280

    if not hidden:
        efn.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
            nn.Sigmoid(),
        )

        # init linear weights
        efn_linear_init(efn.classifier[1])
    else:
        # lr = 7.5e-4
        efn.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, 1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.Sigmoid(),
        )

        efn_linear_init(efn.classifier[1])
        efn_linear_init(efn.classifier[3])

    return efn

def make_efnv2_m_model(num_classes, dropout):
    efn = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

    # dropout = 0.3
    # last_channel = 1280
    # lr = 1.0e-3

    efn.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(1280, num_classes),
        nn.Sigmoid(),
    )

    # init linear weights
    efn_linear_init(efn.classifier[1])

    return efn


def make_tinyvit_model():
    pass



# Define a LightningModule for the classifier
class S2CellClassifierTask(L.LightningModule):
    def __init__(self, model_name, label_mapping, overfit, dropout, learning_rate):
        super().__init__()

        if model_name == "efn_v2_s":
            self.model = make_efnv2_s_model(len(label_mapping), dropout=dropout, hidden=False)
        elif model_name == "efn_v2_s2":
            self.model = make_efnv2_s_model(len(label_mapping), dropout=dropout, hidden=True)
        elif model_name == "efn_v2_m":
            self.model = make_efnv2_m_model(len(label_mapping), dropout=dropout)
        else:
            raise NotImplementedError()

        self.save_hyperparameters("dropout", "learning_rate")
        self.learning_rate = learning_rate

        self.overfit = overfit
        self.label_mapping = label_mapping
        self.s2cell_mapping = s2cell_mapping.S2CellMapping.from_label_mapping(label_mapping)
        num_classes = len(label_mapping)

        # Example input array (for logging graph)
        self.example_input_array = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

        self.subset_accuracy = torchmetrics.classification.MultilabelExactMatch(num_labels=num_classes)
        self.f1_score = torchmetrics.classification.MultilabelF1Score(num_labels=num_classes)
        self.geo_score = geoguessr_score.GeoguessrScore()

    def forward(self, x):
        return self.model(x)

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
        self.log(f"{log_name_prefix}_loss{log_name_suffix}", loss, prog_bar=False, on_step=log_step, on_epoch=(not log_step))

        self.subset_accuracy(z, y)
        self.log(f"{log_name_prefix}_acc{log_name_suffix}", self.subset_accuracy, prog_bar=False, on_step=log_step, on_epoch=(not log_step))

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

        if self.overfit:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=0.5,
                        patience=10,
                    ),
                    "monitor": "val_acc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=0.5,
                        patience=15,
                    ),
                    "monitor": "val_acc",
                    "interval": "step",
                    "frequency": 10000,
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
    print(f"Mode = {args.mode}")

    world1 = World1()
    combined_dataset = Img2LocCombined()
    if args.mode != "overfit":
        #train_dataloader = combined_dataset.train_dataloader()
        #train_dataloader = world1.train_dataloader()
        train_dataloader = combined_dataset.train_dataloader_small()
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

    task = S2CellClassifierTask(
        model_name="efn_v2_s2",
        label_mapping=world1.label_mapping,
        overfit=(args.mode == "overfit"),
        dropout=0.2,
        #learning_rate=7.5e-4,
        learning_rate=1.0e-5,
    )

    if args.mode == "lr_find":
        if args.ckpt_path is not None:
            task = S2CellClassifierTask.load_from_checkpoint(
                args.ckpt_path,
                model_name="efn_v2_s2",
                label_mapping=world1.label_mapping,
                overfit=(args.mode == "overfit"),
            )

        trainer = L.Trainer(
            accelerator=args.accelerator,
            logger=False,
        )
        tuner = lightning.pytorch.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(
            model=task,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        #print(lr_finder.results)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder.png")

        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        return

    #val_check_interval = 5000
    #callback_interval = 5000
    val_check_interval = 10000
    callback_interval = 10000
    stopping_patience = 5 # epochs; TODO
    if args.mode == "overfit":
        val_check_interval = 1.0
        callback_interval = None
        stopping_patience = 30

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_last=True,
            save_top_k=3,
            every_n_train_steps=callback_interval,
        ),
        #L.pytorch.callbacks.EarlyStopping(
        #    patience=stopping_patience,
        #    monitor="val_acc",
        #    mode="max",
        #    verbose=True,
        #),
        L.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step"
        ),
    ]

    limit_args = {}
    if args.limit_batches is not None:
        limit_args = {
            #"limit_train_batches": args.limit_batches,
            "limit_val_batches": args.limit_batches,
        }

    trainer = L.Trainer(
        fast_dev_run=(args.mode == "fast_dev_run"),
        accelerator=args.accelerator,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        logger=L.pytorch.loggers.TensorBoardLogger(
            save_dir=".",
            log_graph=True,
        ),
        **limit_args,
    )

    trainer.fit(
        model=task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.ckpt_path,
    )
    print("Training done")


if __name__ == "__main__":
    main()