from pathlib import Path
import multiprocessing

import torch
import torchvision.transforms.v2 as T
import webdataset as wds
import pandas

import label_mapping

def auto_batch_size():
    DEFAULT = 2
    try:
        dev_name = torch.cuda.get_device_name(0)
        if dev_name == 'NVIDIA A10':
            return 64
        else:
            print("Unknown device:", dev_name)
            return DEFAULT
    except:
        return DEFAULT

def auto_dataloader_workers():
    return (multiprocessing.cpu_count() - 1)

NORMALIZE_T = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_T = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

TRAIN_T = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    NORMALIZE_T,
])

VAL_T = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    NORMALIZE_T,
])


# im2gps 2007 dataset
class Im2gps2007:
    root = Path.home() / "datasets" / "im2gps"
    wds_root = root / "outputs" / "wds"
    
    def __init__(self, s2cell_mapping_name="v1"):
        if s2cell_mapping_name == "v1":
            self.mapping = label_mapping.LabelMapping.read_csv(self.root / "outputs" / "s2cell_2007" / "cells.csv")
            self.annotated_df = pandas.read_pickle(self.root / "outputs" / "s2cell_2007" / "annotated.pkl")
            self.image_id_to_s2cell = {row.id: row.s2cell for row in self.annotated_df.itertuples()}
        else:
            raise NotImplementedError()

    @staticmethod
    def _to_img_latlng(sample):
        img, meta = sample
        label = torch.tensor([meta["latitude"], meta["longitude"]])
        return VAL_T(img), label

    def _make_to_img_label(self, val=True):
        def to_img_label(sample):
            img, meta = sample
            s2cell = self.image_id_to_s2cell.get(meta["id"])
            if s2cell is None:
                return None
            label = self.mapping.get_label(s2cell)
            if val:
                return VAL_T(img), label
            return TRAIN_T(img), label
        return to_img_label

    def _to_label_only(self, sample):
        img, meta = sample
        s2cell = self.image_id_to_s2cell.get(meta["id"])
        if s2cell is None:
            return None
        label = self.mapping.get_label(s2cell)
        return 0, label

    def urls_to_dataset(self, urls, val, shuffle, load_img):
        ds = wds.WebDataset(urls, shardshuffle=shuffle)
        if shuffle:
            ds = ds.shuffle(100)
        if load_img:
            return ds.decode("pil").to_tuple("jpg", "json")\
                .map(self._make_to_img_label(val))\
                .batched(auto_batch_size())
        else:
            return ds.decode(only="json").to_tuple("jpg", "json")\
                .map(self._to_label_only)\
                .batched(auto_batch_size())

    def train_dataloader(self, shuffle=True, load_img=True):
        ds = self.urls_to_dataset(
            str(self.wds_root / "im2gps_2007_train_{000..028}.tar"),
            val=False,
            shuffle=shuffle,
            load_img=load_img,
        )
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def val_dataloader(self, shuffle=True, load_img=True):
        val_dataset = self.urls_to_dataset(
            str(self.wds_root / "im2gps_2007_val_{000..007}.tar"),
            val=True,
            shuffle=shuffle,
            load_img=load_img,
        )
        return wds.WebLoader(val_dataset, batch_size=None, num_workers=auto_dataloader_workers())

    def val_dataloader_latlng(self):
        dataset = wds.WebDataset(str(self.wds_root / "im2gps_2007_val_{000..007}.tar"))
        dataset = dataset.decode("pil").to_tuple("jpg", "json")\
            .map(self._to_img_latlng)\
            .batched(auto_batch_size())
        return wds.WebLoader(dataset, batch_size=None, num_workers=auto_dataloader_workers())


# im2gps test sets
class Im2gpsTest:
    root_3k = Path.home() / "datasets" / "im2gps3ktest"

    @classmethod
    def test_dataloader_3k(cls, batch_size=1):
        test_dataset = wds.WebDataset(str(cls.root_3k / "wds" / "im2gps3ktest_000.tar"))
        test_dataset = test_dataset.decode("pil").to_tuple("jpg", "json")\
            .map(Im2gps2007._to_img_latlng)\
            .batched(batch_size)
        return wds.WebLoader(test_dataset, batch_size=None, num_workers=auto_dataloader_workers())


if __name__ == "__main__":
    # Test datasets
    for inputs, targets in Im2gps2007("v1").train_dataloader():
        print(inputs.shape, targets.shape)
        break
    for inputs, targets in Im2gps2007("v1").val_dataloader():
        print(inputs.shape, targets.shape)
        break
    for inputs, targets in Im2gps2007("v1").val_dataloader_latlng():
        print(inputs.shape, targets.shape)
        break
    for inputs, targets in Im2gpsTest.test_dataloader_3k():
        print(inputs.shape, targets.shape)
        break