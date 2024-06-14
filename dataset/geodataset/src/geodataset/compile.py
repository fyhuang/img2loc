# Compile a webdataset from a directory of contents (images, labels, etc.)

import sys
import argparse
import json

from pathlib import Path

import pandas
import webdataset
import tqdm


def write_dataset_as_wds(dataset_df, img_base_dir, out_pattern, out_missing):
    def write_wds_row(row, sink):
        full_img_path = img_base_dir / row.img_path
        if not full_img_path.exists():
            out_missing.append(full_img_path)
            return

        wds_object = {
            "__key__": "{:09d}".format(row.Index),
            "jpg": full_img_path.read_bytes(),
            "json": json.dumps(row._asdict()).encode("utf-8"),
        }
        sink.write(wds_object)

    dataset_df = dataset_df.sample(frac=1) # shuffle

    with webdataset.ShardWriter(out_pattern, encoder=False) as sink:
        for row in tqdm.tqdm(dataset_df.itertuples(), total=len(dataset_df.index)):
            write_wds_row(row, sink)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="File containing DataFrame")
    parser.add_argument("--images", help="Directory containing images")
    parser.add_argument("--output", help="Output tar pattern (e.g. \"data_%06d.tar\")")
    args = parser.parse_args()

    # TODO(fyhuang): support more formats
    dataset_df = pandas.read_pickle(args.input)

    out_missing = []
    write_dataset_as_wds(dataset_df, Path(args.images), args.output, out_missing)

    if len(out_missing) > 0:
        print(f"Couldn't find {len(out_missing)} images")


if __name__ == "__main__":
    main()