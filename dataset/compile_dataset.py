# Compile a webdataset from a directory of contents (images, labels, etc.)

import sys
import argparse
import tarfile

from pathlib import Path

import check_dataset


def get_sorted_filenames(dirpath):
    raw_names = []
    for child in dirpath.iterdir():
        if child.is_file():
            raw_names.append(str(child))
    
    raw_names.sort()
    return raw_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Directory containing dataset contents")
    parser.add_argument("output_tar", help="Output tar file")
    args = parser.parse_args()

    filenames = get_sorted_filenames(Path(args.dataset_dir))
    if check_dataset.check_filename_list(filenames) == False:
        print("Dataset is not valid")
        sys.exit(1)

    # Add files to a tar file
    with tarfile.open(args.output_tar, "w") as tf:
        for filename in filenames:
            tf.add(filename, recursive=False)


if __name__ == "__main__":
    main()