"""Utility for downloading images in parallel.
"""

import argparse
from pathlib import Path

import requests
import pandas
import tqdm

from . import rate_limited_parallel

class Downloader:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def prepare_download(self, row):
        out_path = self.output_dir / row.img_path
        if out_path.is_file():
            if out_path.stat().st_size > 0:
                # Already downloaded, skip
                return None

        out_subdir = out_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        return out_path

    def download_one(self, out_path, url):
        """Download one image.
        """
        with requests.get(url) as r:
            r.raise_for_status()

            # From https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=512*1024):
                    f.write(chunk)

    def download_all(self, df, rate_limit):
        parallel = rate_limited_parallel.RateLimitedExecutor(rate_limit)

        try:
            futures = []
            for row in tqdm.tqdm(df.itertuples()):
                out_path = self.prepare_download(row)
                if out_path is None:
                    # Already downloaded, skip
                    continue
                futures.append(parallel.submit(self.download_one, out_path, row.url))

            print("Submitted all futures")
            for future in tqdm.tqdm(futures):
                future.result()
        finally:
            parallel.shutdown()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save images to")
    parser.add_argument("--rate_limit", type=float, default=5.0, help="Rate limit in images per second")
    parser.add_argument("dataframe_path", help="Pickled dataframe with columns 'img_path' and 'url'")
    args = parser.parse_args()

    dataset_df = pandas.read_pickle(args.dataframe_path)
    d = Downloader(args.output_dir)
    d.download_all(dataset_df, args.rate_limit)


if __name__ == "__main__":
    main()