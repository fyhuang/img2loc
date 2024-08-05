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

from mlutil import label_mapping, s2cell_mapping, geoguessr_score, hier_s2_classifier
from datasets import World1, Img2LocCombined, Im2gps2007

import timm
#from tiny_vit import tiny_vit_21m_224


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

def make_efnv2_s_model(num_classes, dropout, hidden, dropout2=None):
    efn = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # dropout = 0.2
    # norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
    # last_channel = 1280

    if not hidden:
        efn.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
            #nn.Sigmoid(),
        )

        # init linear weights
        efn_linear_init(efn.classifier[1])
    else:
        # lr = 7.5e-4
        print(f"Hidden size = {hidden}, dropout = {dropout}, dropout2 = {dropout2}")
        layers = [
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, hidden),
        ]

        if dropout2 is not None:
            layers.append(nn.Dropout(p=dropout2, inplace=True))

        layers += [
            nn.SiLU(inplace=True),
            nn.Linear(hidden, num_classes),
        ]

        efn.classifier = nn.Sequential(
            *layers
            #nn.Sigmoid(),
        )

        efn_linear_init(efn.classifier[1])
        efn_linear_init(efn.classifier[-1])

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


def make_vitb16_model(num_classes):
    vitb = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # dropout = 0.0
    # hidden_dim = 768
    # mlp_dim = 3072
    # lr = 3e-4

    pre_logits_size = 2048
    heads_layers = collections.OrderedDict()
    heads_layers["pre_logits"] = nn.Linear(768, pre_logits_size)
    heads_layers["act"] = nn.Tanh()
    heads_layers["head"] = nn.Linear(pre_logits_size, num_classes)
    vitb.heads = nn.Sequential(heads_layers)

    fan_in = vitb.heads.pre_logits.in_features
    nn.init.trunc_normal_(vitb.heads.pre_logits.weight, std=math.sqrt(1.0 / fan_in))
    nn.init.zeros_(vitb.heads.pre_logits.bias)

    nn.init.zeros_(vitb.heads.head.weight)
    nn.init.zeros_(vitb.heads.head.bias)

    return vitb

def make_tinyvit_21m224_model(num_classes, dropout, hidden):
    tvit = tiny_vit_21m_224(pretrained=True)

    # embed_dims = [96, 192, 384, 576]
    # depths = [2, 2, 6, 2]
    # drop_path_rate = 0.2
    # layer_lr_decay = 1.0

    # LRs at configs:
    # https://github.com/wkcn/TinyViT/blob/main/configs/22kto1k/tiny_vit_21m_22kto1k.yaml
    # https://github.com/wkcn/TinyViT/blob/main/configs/higher_resolution/tiny_vit_21m_224to384.yaml
    # https://github.com/wkcn/TinyViT/blob/main/configs/higher_resolution/tiny_vit_21m_384to512.yaml

    def _set_lr_scale(m, scale):
        for p in m.parameters():
            p.lr_scale = scale

    decay_rate = 1.0
    depths_sum = sum([2, 2, 6, 2])
    last_lr_scale = decay_rate ** (depths_sum - 3 - 1)

    if hidden:
        classifier = nn.Sequential(
            nn.Linear(576, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden, num_classes),
        )

        nn.init.trunc_normal_(classifier[0].weight, std=0.02)
        nn.init.zeros_(classifier[0].bias)

        nn.init.trunc_normal_(classifier[-1].weight, std=0.02)
        nn.init.zeros_(classifier[-1].bias)

        classifier[0].apply(lambda m: _set_lr_scale(m, last_lr_scale))
        classifier[-1].apply(lambda m: _set_lr_scale(m, last_lr_scale))

        tvit.head = classifier
    else:
        tvit.head = nn.Linear(576, num_classes)
        nn.init.trunc_normal_(tvit.head.weight, std=0.02)
        nn.init.zeros_(tvit.head.bias)

        tvit.head.apply(lambda m: _set_lr_scale(m, last_lr_scale))

    #def _check_lr_scale(m):
    #    for p in m.parameters():
    #        assert hasattr(p, 'lr_scale'), p.param_name
    #tvit.apply(_check_lr_scale)

    def _enable_grad(m):
        for p in m.parameters():
            p.requires_grad = True
    tvit.apply(_enable_grad)

    return tvit

def make_tinyvit_21m224_timm_model(num_classes, dropout, hidden, hier=False):
    tvit = timm.create_model(
        "hf-hub:timm/tiny_vit_21m_224.dist_in22k",
        pretrained=True,
        num_classes=num_classes,
    )

    if hidden is not None:
        from functools import partial
        tvit.head = timm.layers.NormMlpClassifierHead(
            in_features=576,
            num_classes=num_classes,
            hidden_size=hidden,
            drop_rate=dropout,
            pool_type="avg",
            norm_layer=partial(timm.layers.LayerNorm2d, eps=1e-5),
        )

    if hier:
        tvit.head = hier_s2_classifier.HierS2ClassifierHead(576, s2cell_mapping.S2CellMapping.from_label_mapping(label_mapping))
        
    return tvit

def make_mnet3_model(num_classes, dropout):
    mnet3 = models.mobilenet_v3_large(weights="IMAGENET1K_V2")

    hidden_size = 2048
    mnet3.classifier = nn.Sequential(
        nn.Linear(mnet3.classifier[0].in_features, hidden_size),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(hidden_size, num_classes), # out is 1776
    )

    torch.nn.init.xavier_uniform_(mnet3.classifier[0].weight)
    torch.nn.init.xavier_uniform_(mnet3.classifier[3].weight)

    return mnet3



# Define a LightningModule for the classifier
class S2CellClassifierTask(L.LightningModule):
    def __init__(self, model_name, label_mapping, overfit, dropout, learning_rate, export=False):
        super().__init__()

        if model_name == "efn_v2_s":
            self.model = make_efnv2_s_model(len(label_mapping), dropout=dropout, hidden=False)
        elif model_name == "efn_v2_s2":
            self.model = make_efnv2_s_model(len(label_mapping), dropout=dropout, hidden=1024)
        elif model_name == "efn_v2_s3":
            self.model = make_efnv2_s_model(len(label_mapping), dropout=dropout, hidden=2048)
        elif model_name == "efn_v2_s4":
            self.model = make_efnv2_s_model(len(label_mapping), dropout=0.0, hidden=2048, dropout2=dropout)
        elif model_name == "efn_v2_m":
            self.model = make_efnv2_m_model(len(label_mapping), dropout=dropout)
        elif model_name == "vitb16":
            self.model = make_vitb16_model(len(label_mapping))
        elif model_name == "tinyvit_21m_224":
            self.model = make_tinyvit_21m224_timm_model(len(label_mapping), dropout=0.0, hidden=None)
        elif model_name == "tinyvit_21m_224_v2":
            self.model = make_tinyvit_21m224_timm_model(len(label_mapping), dropout=dropout, hidden=2048)
        elif model_name == "tinyvit_21m_224_v3":
            self.model = make_tinyvit_21m224_timm_model(len(label_mapping), dropout=dropout, hidden=None, hier=True)
        elif model_name == "mnet3":
            self.model = make_mnet3_model(len(label_mapping), dropout=dropout)
        elif model_name == "resnet50.a1_in1k":
            self.model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=len(label_mapping))
        else:
            raise NotImplementedError(f"No model: {model_name}")

        print(f"Model: {model_name}")
        self.save_hyperparameters("dropout", "learning_rate")
        self.learning_rate = learning_rate

        self.overfit = overfit
        self.label_mapping = label_mapping
        self.s2cell_mapping = s2cell_mapping.S2CellMapping.from_label_mapping(label_mapping)
        num_classes = len(label_mapping)
        self.export = export

        # Example input array (for logging graph)
        self.example_input_array = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

        self.subset_accuracy = torchmetrics.classification.MultilabelExactMatch(num_labels=num_classes)
        self.f1_score = torchmetrics.classification.MultilabelF1Score(num_labels=num_classes)
        self.geo_score = geoguessr_score.GeoguessrScore()

    def forward(self, x):
        y = self.model(x)
        if self.export:
            y = torch.sigmoid(y)
        return y

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

        # Positive weights for BCEWithLogitsLoss
        # Imbalance is ~1:155
        positive_weights = torch.ones(y.shape[1], dtype=torch.float32, device=y.device) * 150.0

        logits = self.forward(x)
        loss = nn.BCEWithLogitsLoss(pos_weight=positive_weights)(logits, y)
        self.log(f"{log_name_prefix}_loss{log_name_suffix}", loss, prog_bar=False, on_step=log_step, on_epoch=(not log_step))

        z = nn.Sigmoid()(logits)

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
        #optimizer = optim.Adam(
        #    self.parameters(),
        #    lr=self.learning_rate,
        #)
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            #weight_decay=0.05,
            weight_decay=1e-8,
        )

        # For ~30 epochs
        #scheduler = optim.lr_scheduler.SequentialLR(
        #    optimizer,
        #    schedulers=[
        #        optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5),
        #        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6),
        #        optim.lr_scheduler.ConstantLR(optimizer, factor=1e-6, total_iters=float('inf')),
        #    ],
        #    milestones=[5, 30],
        #)

        # For ~15 epochs (w+20%)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=13, eta_min=1e-6),
                optim.lr_scheduler.ConstantLR(optimizer, factor=1e-6, total_iters=float('inf')),
            ],
            milestones=[3, 16],
        )


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                #"scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                #    optimizer,
                #    mode="min",
                #    factor=0.5,
                #    patience=50,
                #),
                #"interval": "step",
                #"frequency": 1000,
                #"monitor": "train_loss_step",

                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--limit_batches", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)

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
        train_dataloader = combined_dataset.train_dataloader(subset=4)
        #train_dataloader = world1.train_dataloader()
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

    model_hparams_efn_v2_s2 = {
        "model_name": "efn_v2_s2",
        "dropout": 0.2,
        "learning_rate": 1.0e-3,
    }

    model_hparams_efn_v2_s3 = {
        "model_name": "efn_v2_s3",
        "dropout": 0.2,
        "learning_rate": 1.0e-3,
    }

    model_hparams_efn_v2_s4 = {
        "model_name": "efn_v2_s4",
        "dropout": 0.2,
        "learning_rate": 1.0e-3,
    }

    model_hparams_vitb16 = {
        "model_name": "vitb16",
        "dropout": 0.0,
        "learning_rate": 3.0e-4,
    }

    model_hparams_tinyvit_21m_224 = {
        "model_name": "tinyvit_21m_224",
        "dropout": 0.0,
        "learning_rate": 5.0e-4,
        #"learning_rate": 1.0e-3,
        #"learning_rate": 1.0e-4,
    }

    model_hparams_tinyvit_21m_224_v2 = {
        "model_name": "tinyvit_21m_224_v2",
        "dropout": 0.2,
        #"learning_rate": 1.0e-3,
        "learning_rate": 5.0e-4,
    }

    model_hparams_mnet3 = {
        "model_name": "mnet3",
        "dropout": 0.2,
        "learning_rate": 1e-3,
    }

    task = S2CellClassifierTask(
        label_mapping=world1.label_mapping,
        overfit=(args.mode == "overfit"),
        #model_name="resnet50.a1_in1k", dropout=0.0, learning_rate=1e-3,
        **model_hparams_tinyvit_21m_224,
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
    val_check_interval = 2500
    callback_interval = 2500
    if args.mode == "overfit":
        val_check_interval = 1000
        callback_interval = 1000

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_last=True,
            save_top_k=3,
            every_n_train_steps=callback_interval,
        ),
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
        check_val_every_n_epoch=None,
        max_steps=args.max_steps,
        logger=L.pytorch.loggers.TensorBoardLogger(
            save_dir=".",
            #log_graph=True,
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