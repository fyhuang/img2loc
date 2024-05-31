import time
import datetime
import json
import io
import concurrent.futures
import threading
import queue

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas
import requests
import flickrapi
import tqdm
import tqdm.notebook

class _QueryTag(NamedTuple):
    tag: str
    query_str: str

def _read_queries(queries_file_path):
    def read_query_lines():
        with queries_file_path.open(encoding="latin_1") as f:
            for raw_line in f.readlines():
                line = raw_line.strip()
                if len(line) < 1:
                    continue
                if line[0] == '#':
                    continue
                yield line

    # First read all negative queries
    neg_queries = []
    for line in read_query_lines():
        if line[0] == '-':
            neg_queries.append(line)
    neg_query_string = " ".join(neg_queries)

    # Generate list of queries
    all_queries = []
    for line in read_query_lines():
        if line[0] == '-':
            continue
        all_queries.append(
            _QueryTag(
                tag=line,
                query_str=f"{line} {neg_query_string}"
            )
        )

    return all_queries


class Im2GpsConfig(NamedTuple):
    min_block_photos: int
    initial_block_len_secs: int
    search_start_timestamp: int
    search_end_timestamp: int


class Im2GpsDatasetCreator:
    def __init__(
            self,
            flickr_api_key: tuple[str, str],
            queries_file_path: Path,
            search_config: Im2GpsConfig,
        ):
        # Pass store_token=False to work around https://github.com/sybrenstuvel/flickrapi/issues/75
        # (We don't need user auth anyways)
        self.fapi = flickrapi.FlickrAPI(flickr_api_key[0], flickr_api_key[1], format="parsed-json", store_token=False)

        # Enforce the Flickr API rate limit of 1 QPS
        self.last_api_time: Optional[float] = None

        self.all_queries = _read_queries(queries_file_path)

        self.search_config = search_config

    def search_time_range(self, query_str, min_date: int, max_date: int, page: int = 1):
        # min_date and max_date are Unix timestamps

        # Retry a few times in case of 500
        for retry_number in range(3):
            # Limit API calls to 1 QPS with generous buffer
            start_time = time.time()
            if self.last_api_time is not None:
                wait_time = max(0, 1.5 - (start_time - self.last_api_time))
                time.sleep(wait_time)
            self.last_api_time = start_time

            try:
                return self.fapi.photos.search(
                    privacy_filter="1", # public only
                    text=query_str,
                    media="photos",
                    has_geo="1",
                    accuracy="6", # region level
                    min_upload_date=min_date,
                    max_upload_date=max_date,
                    sort="interestingness-desc",
                    page=page,
                    per_page=250,
                    extras="geo,date_upload"
                )
            except flickrapi.FlickrError as e:
                if retry_number == 2:
                    raise
                time.sleep(5)
                continue

    def query_save_one_tag(self, qt: _QueryTag, results_path: Path):
        block_start = self.search_config.search_start_timestamp
        block_len = self.search_config.initial_block_len_secs

        # Skip the query if we already have results
        df_out_path = results_path / f"{qt.tag}.pkl"
        if df_out_path.exists():
            print(f"{qt.tag}: already exists, skipping")
            return

        results_df = pandas.DataFrame()

        try:
            while block_start < self.search_config.search_end_timestamp:
                results = self.search_time_range(qt.query_str, block_start, block_start + block_len - 1)
                total_results = results["photos"]["total"] 
                #print(f"  {qt.tag}: block {block_start}->{block_start+block_len} total {total_results}")
                if (
                    total_results < self.search_config.min_block_photos and
                    block_start + block_len < self.search_config.search_end_timestamp
                ):
                    block_len = int(block_len * 2)
                    continue

                # Found a good block_len
                print(f"{qt.tag}: block {block_start}->{block_start+block_len} total {total_results}")

                # Read all of the pages
                for page_num in range(1, results["photos"]["pages"]):
                    if page_num > 1:
                        #print(f"  {qt.tag}: page {page_num}")
                        results = self.search_time_range(qt.query_str, block_start, block_start + block_len - 1, page=page_num)
                    photos_json = io.StringIO(json.dumps(results["photos"]["photo"]))
                    df = pandas.read_json(photos_json, orient="records")
                    if len(df.index) == 0:
                        continue

                    # Fix dtypes
                    if df["server"].dtype != np.int64:
                        df["server"] = df["server"].astype(np.int64)
                    df["interestingness"] = df.index.to_numpy() + (page_num - 1) * 250
                    results_df = pandas.concat([results_df, df], ignore_index=True)

                block_start += block_len
                block_len = self.search_config.initial_block_len_secs
        except flickrapi.FlickrError as e:
            print(f"{qt.tag}: error: {e}")
            return

        # Save results
        results_df.to_pickle(df_out_path)
        print(f"{qt.tag}: saved to {df_out_path}")

    def query_all(self, results_path: Path):
        for qt in self.all_queries:
            self.query_save_one_tag(qt, results_path)

    def assemble(self, query_results_path: Path):
        # Assemble all the results into a single dataframe
        dataset_df = pandas.DataFrame()
        for qt in self.all_queries:
            out_path = query_results_path / f"{qt.tag}.pkl"
            if out_path.exists():
                query_df = pandas.read_pickle(out_path)
                if len(query_df.index) == 0:
                    continue
                query_df["tag"] = qt.tag
                dataset_df = pandas.concat([dataset_df, query_df], ignore_index=True)
            else:
                print(f"No file {out_path}")

        dataset_df.drop_duplicates(subset=["id", "owner", "secret", "server"], inplace=True)
        return dataset_df.sample(frac=1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_path", type=str, required=True, help="Flickr API key")
    parser.add_argument("--queries_file_path", type=str, required=True, help="File with queries")
    parser.add_argument("--results_path", type=str, required=True, help="Path to save results")
    args = parser.parse_args()

    # Command line tool defaults to a pre-defined config
    config = Im2GpsConfig(
        min_block_photos=150,
        initial_block_len_secs=4 * 60 * 60, # 4 hours
        search_start_timestamp=1672531200, # 2023-01-01T00:00:00Z,
        search_end_timestamp=1704067199, # 2023-12-31T23:59:59Z
    )

    api_key, api_secret = \
        [l.strip() for l in Path(args.api_key_path).read_text().splitlines()]

    creator = Im2GpsDatasetCreator(
        flickr_api_key=(api_key, api_secret),
        queries_file_path=Path(args.queries_file_path),
        search_config=config
    )

    creator.query_all(Path(args.results_path))

    # TODO(fyhuang): specify the filename
    creator.assemble(Path(args.results_path)).to_pickle("all.pkl")

if __name__ == "__main__":
    main()