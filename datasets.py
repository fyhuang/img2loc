from pathlib import Path
from braceexpand import braceexpand
import multiprocessing

import torch
import torchvision.transforms.v2 as T
import webdataset as wds
import pandas

from mlutil import label_mapping, s2cell_mapping

def auto_batch_size():
    DEFAULT = 16
    try:
        dev_name = torch.cuda.get_device_name(0)
        if dev_name == 'NVIDIA A10':
            return 64
        else:
            print("auto_batch_size: unknown device", dev_name)
            return DEFAULT
    except:
        return DEFAULT

def auto_dataloader_workers():
    return (multiprocessing.cpu_count() - 1)

def auto_shuffle_size():
    DEFAULT = 1000
    try:
        dev_name = torch.cuda.get_device_name(0)
        if dev_name == 'NVIDIA A10':
            return 100_000
        else:
            print("auto_batch_size: unknown device", dev_name)
            return DEFAULT
    except:
        return DEFAULT

NORMALIZE_T = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
NORM_HALF_T = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
UNNORMALIZE_T = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

JITTER_FULL_T = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
JITTER_BC_T = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.01, hue=0.01)
JITTER_LESS_T = T.ColorJitter(brightness=0.01, contrast=0.01, saturation=0, hue=0)

IMG_CROP_SIZE = 224
#IMG_CROP_SIZE = 384

TRAIN_T = T.Compose([
    T.RandomResizedCrop(IMG_CROP_SIZE),
    T.RandomHorizontalFlip(),
    T.ToImage(),
    JITTER_LESS_T,
    T.ToDtype(torch.float32, scale=True),
    NORMALIZE_T,
    #NORM_HALF_T,
])

VAL_T = T.Compose([
    T.Resize(IMG_CROP_SIZE),
    T.CenterCrop(IMG_CROP_SIZE),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    NORMALIZE_T,
    #NORM_HALF_T,
])

VAL_ZOOM_T = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    NORMALIZE_T,
])


class _GeoDatasetTransformer:
    def __init__(self, label_mapping, s2cell_mapping):
        self.label_mapping = label_mapping
        self.s2cell_mapping = s2cell_mapping

    def meta_to_label_tensor(self, meta):
        # Compute the label for multi class classification
        single_token = self.s2cell_mapping.lat_lng_to_token(meta["latitude"], meta["longitude"])
        if single_token is None:
            return None
        single_label = self.label_mapping.get_label(single_token)

        # Compute the label for multilabel classification
        multihot_list = self.s2cell_mapping.lat_lng_to_multihot_list(meta["latitude"], meta["longitude"])

        label_tensor = torch.cat((
            torch.tensor([meta["latitude"], meta["longitude"]]),
            torch.tensor([single_label]),
            torch.tensor(multihot_list),
        ))

        return label_tensor

    def make_to_img_label(self, val=True):
        def to_img_label(sample):
            img, meta = sample
            transformed_img = VAL_T(img) if val else TRAIN_T(img)
            #transformed_img = VAL_ZOOM_T(img) if val else TRAIN_T(img)
            labels = self.meta_to_label_tensor(meta)
            if labels is None:
                return None
            return transformed_img, labels
        return to_img_label

    def to_label_only(self, sample):
        img, meta = sample
        return 0, self.meta_to_label_tensor(meta)


def urls_to_dataset(urls, transformer, val, shuffle, load_img):
    if type(urls) == list:
        # Do manual brace expansion
        expanded_urls = []
        for url_template in urls:
            expanded_urls.extend(braceexpand(url_template))
        urls = expanded_urls

    ds = wds.WebDataset(urls, shardshuffle=shuffle)
    if shuffle:
        ds = ds.shuffle(auto_shuffle_size())
    if load_img:
        return ds.decode("pil").to_tuple("jpg", "json")\
            .map(transformer.make_to_img_label(val))\
            .batched(auto_batch_size())
    else:
        return ds.decode(only="json").to_tuple("jpg", "json")\
            .map(transformer.to_label_only)\
            .batched(auto_batch_size())




# im2gps 2007 dataset
class Im2gps2007:
    root = Path.home() / "datasets/img2loc"
    wds_root = root / "im2gps_2007"

    overfit_wds = Path.home() / "datasets" / "im2gps_overfit" / "wds"
    
    def __init__(self, s2cell_mapping_name="v2"):
        if s2cell_mapping_name == "v1":
            self.mapping = label_mapping.LabelMapping.read_csv(self.root / "outputs" / "s2cell_2007" / "cells.csv")
            self.annotated_df = pandas.read_pickle(self.root / "outputs" / "s2cell_2007" / "annotated.pkl")
            self.image_id_to_s2cell = {row.id: row.s2cell for row in self.annotated_df.itertuples()}
        elif s2cell_mapping_name == "v2":
            self.mapping = label_mapping.LabelMapping.read_csv(self.root / "s2cell_930_ml.csv")
            self.transformer = _GeoDatasetTransformer(self.mapping, s2cell_mapping.S2CellMapping.from_label_mapping(self.mapping))
        else:
            raise NotImplementedError()

    def _make_to_img_label(self, val=True):
        #def to_img_label(sample):
        #    img, meta = sample
        #    if val:
        #        return VAL_T(img), self.transformer.meta_to_label_tensor(meta)
        #    return TRAIN_T(img), self.transformer.meta_to_label_tensor(meta)
        #return to_img_label
        return self.transformer.make_to_img_label(val)

    def _to_label_only(self, sample):
        img, meta = sample
        return 0, self.transformer.meta_to_label_tensor(meta)

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

    def train_dataset(self, shuffle=True, load_img=True):
        # Total examples: 472k
        return self.urls_to_dataset(
            str(self.wds_root / "im2gps_2007_train_{000..028}.tar"),
            val=False,
            shuffle=shuffle,
            load_img=load_img,
        )
    
    def train_dataloader(self, *args, **kwargs):
        ds = self.train_dataset(*args, **kwargs)
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def val_dataset(self, shuffle=True, load_img=True):
        # Total examples: 118k
        val_dataset = self.urls_to_dataset(
            str(self.wds_root / "im2gps_2007_val_{000..007}.tar"),
            val=True,
            shuffle=shuffle,
            load_img=load_img,
        )
        return wds.WebLoader(val_dataset, batch_size=None, num_workers=auto_dataloader_workers())

    def val_dataloader(self, *args, **kwargs):
        ds = self.val_dataset(*args, **kwargs)
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def val_dataloader_latlng(self):
        dataset = wds.WebDataset(str(self.wds_root / "im2gps_2007_val_{000..007}.tar"))
        dataset = dataset.decode("pil").to_tuple("jpg", "json")\
            .map(self._to_img_latlng)\
            .batched(auto_batch_size())
        return wds.WebLoader(dataset, batch_size=None, num_workers=auto_dataloader_workers())

    def overfit_dataloader_one(self, val=False):
        ds = self.urls_to_dataset(
            str(self.overfit_wds / "im2gps_overfit_one_000.tar"),
            val=False,
            shuffle=True,
            load_img=True
        )
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def overfit_dataloader_five(self, val=False):
        ds = self.urls_to_dataset(
            str(self.overfit_wds / "im2gps_overfit_five_000.tar"),
            val=False,
            shuffle=True,
            load_img=True
        )
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def test_dataloader_3k(self, batch_size=1):
        test_dataset = wds.WebDataset(str(self.root / "im2gps3ktest/im2gps3ktest_000.tar"))
        test_dataset = test_dataset.decode("pil").to_tuple("jpg", "json")\
            .map(self._make_to_img_label(val=True))\
            .batched(batch_size)
        return wds.WebLoader(test_dataset, batch_size=None, num_workers=auto_dataloader_workers())


class World1:
    root = Path.home() / "datasets/img2loc"
    wds_root = root / "world1"

    overfit_wds = root / "world1_overfit"
    
    def __init__(self, s2cell_mapping_name="s2cell_930_ml"):
        self.label_mapping = label_mapping.LabelMapping.read_csv(self.root / f"{s2cell_mapping_name}.csv")
        self.s2cell_mapping = s2cell_mapping.S2CellMapping.from_label_mapping(self.label_mapping)

        self.transformer = _GeoDatasetTransformer(self.label_mapping, self.s2cell_mapping)

    def train_dataset(self, shuffle=True, load_img=True):
        # Total examples: 40k
        return urls_to_dataset(
            [str(self.wds_root / "world1_{000..001}.tar")] * 3,
            self.transformer,
            val=False,
            shuffle=shuffle,
            load_img=load_img,
        )

    def train_dataloader(self, *args, **kwargs):
        ds = self.train_dataset(*args, **kwargs)
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def overfit_dataloader_one(self, train_repeat=3, val=False):
        urls = [str(self.overfit_wds / "world1_overfit_one_000.tar")]
        if not val:
            urls = urls * train_repeat
        ds = urls_to_dataset(
            urls,
            self.transformer,
            val=val,
            shuffle=True,
            load_img=True
        )
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def overfit_dataloader_five(self, train_repeat=3, val=False):
        urls = [str(self.overfit_wds / "world1_overfit_five_000.tar")]
        if not val:
            urls = urls * train_repeat
        ds = urls_to_dataset(
            urls,
            self.transformer,
            val=val,
            shuffle=True,
            load_img=True
        )
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())


class Img2LocCombined:
    """Combined dataset of im2gps 2007, 2023, and world1"""
    root = Path.home() / "datasets/img2loc"

    def __init__(self, s2cell_mapping_name="s2cell_930_ml"):
        mapping = label_mapping.LabelMapping.read_csv(self.root / f"{s2cell_mapping_name}.csv")
        self.transformer = _GeoDatasetTransformer(mapping, s2cell_mapping.S2CellMapping.from_label_mapping(mapping))

    def train_dataloader(self, subset=0):
        # Total is ~800k examples
        if subset == 0:
            urls = [
                # im2gps v2 is ~773k examples
                str(self.root / "im2gps_v2/im2gps_v2_train_{000..040}.tar"),
                # ~38k examples
                str(self.root / "world1/world1_train_{000..001}.tar"),
            ]
        elif subset == 1:
            # only world
            urls = str(self.root / "world1/world1_train_{000..001}.tar")
        elif subset == 2:
            # world + ~10% of im2gps
            urls = [
                str(self.root / "world1/world1_train_{000..001}.tar"),
                str(self.root / "im2gps_v2/im2gps_v2_train_{000..004}.tar"),
            ]
        elif subset == 3:
            # world + 50% of im2gps
            urls = [
                str(self.root / "world1/world1_train_{000..001}.tar"),
                str(self.root / "im2gps_v2/im2gps_v2_train_{000..019}.tar"),
            ]

        ds = urls_to_dataset(urls, self.transformer, val=False, shuffle=True, load_img=True)
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())

    def val_dataloader(self):
        urls = [
            str(self.root / "im2gps_v2/im2gps_v2_val_000.tar"),
            str(self.root / "world1/world1_val_000.tar"),
        ]
        ds = urls_to_dataset(urls, self.transformer, val=True, shuffle=True, load_img=True)
        return wds.WebLoader(ds, batch_size=None, num_workers=auto_dataloader_workers())


if __name__ == "__main__":
    # Test datasets
    #for inputs, targets in Im2gps2007("v1").train_dataloader():
    #    print("Im2gps2007::train", inputs.shape, targets.shape)
    #    break
    #for inputs, targets in Im2gps2007("v1").val_dataloader():
    #    print("Im2gps2007::val", inputs.shape, targets.shape)
    #    break
    #for inputs, targets in Im2gps2007("v1").val_dataloader_latlng():
    #    print("Im2gps2007::val_ll", inputs.shape, targets.shape)
    #    break
    #for inputs, targets in Im2gpsTest.test_dataloader_3k():
    #    print("Im2gpsTest", inputs.shape, targets.shape)
    #    break
    for inputs, targets in Img2LocCombined().train_dataloader():
        print("Img2LocCombined::train", inputs.shape, targets.shape)
        break
    for inputs, targets in Img2LocCombined().val_dataloader():
        print("Img2LocCombined::val", inputs.shape, targets.shape)
        break