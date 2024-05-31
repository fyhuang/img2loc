"""Utilities for creating a StreetView dataset.

StreetView datasets have the following stages:

Stage 1: raw
         Raw points in a DataFrame with columns "lat" and "lng".
Stage 2: sv_cleaned
         Transform raw points into closest SV panoramas. Filter out non-existent points
Stage 3: parameterized
         Select heading, pitch, etc. for each panorama.
Stage 4: final
         Remove points that are not in the S2 cell mapping (no point in downloading them).

"""

import hashlib
import hmac
import base64
import urllib.parse
import time
from typing import NamedTuple, Optional
from pathlib import Path

import pandas
import numpy

import requests
import tqdm

from . import rate_limited_parallel

class _SvResult(NamedTuple):
    key: int
    status: str
    lat: float
    lng: float
    pano_id: str

class _SvApiError(ValueError):
    pass

class StreetViewDatasetCreator:
    def __init__(self, api_key, sign_secret):
        self.api_key = api_key
        self.sign_secret = sign_secret

    def _sign_url(self, url_string):
        # Sign the URL
        url = urllib.parse.urlparse(url_string)
        url_to_sign = url.path + "?" + url.query
        decoded_key = base64.urlsafe_b64decode(self.sign_secret)
        signature = hmac.new(decoded_key, url_to_sign.encode(), hashlib.sha1)
        encoded_signature = base64.urlsafe_b64encode(signature.digest())
        return url_string + "&signature=" + encoded_signature.decode()

    def annotate_one(self, key, row):
        for _ in range(10):
            # Call the Street View API to get the nearest panorama
            response = requests.get(self._sign_url(
                "https://maps.googleapis.com/maps/api/streetview/metadata"
                f"?location={row.lat},{row.lng}"
                f"&radius=100"
                f"&key={self.api_key}"
            ))

            response.raise_for_status()
            result = response.json()

            sv_status = (result["status"])
            if sv_status == "UNKNOWN_ERROR":
                # According to API docs, retrying can help
                time.sleep(5)
                continue

            if sv_status not in (
                "OK",
                "ZERO_RESULTS",
                "NOT_FOUND",
            ):
                raise _SvApiError(f"{sv_status}")

            break
        else:
            print(f"Failed to get a valid response: {row.lat}, {row.lng}")

        location = result.get("location", {})
        sv_lat = (location.get("lat"))
        sv_lng = (location.get("lng"))
        sv_pano = (result.get("pano_id"))

        return _SvResult(
            key=key,
            status=sv_status,
            lat=sv_lat,
            lng=sv_lng,
            pano_id=sv_pano,
        )

    def annotate(
            self,
            raw_df,
            # Street View allows 30k QPM == 500 QPS. 250 QPS is 1/2 of that.
            rate_limit=100.0,
            ):

        out_df = raw_df.rename(
            columns={"lat": "raw_lat", "lng": "raw_lng"},
            inplace=False,
        )

        sv_lat = pandas.Series(index=raw_df.index, dtype=float)
        sv_lng = pandas.Series(index=raw_df.index, dtype=float)
        sv_status = pandas.Series(index=raw_df.index, dtype=str)
        sv_pano_id = pandas.Series(index=raw_df.index, dtype=str)

        print(f"Rate limit = {rate_limit} QPS")
        parallel = rate_limited_parallel.RateLimitedExecutor(rate_limit, max_workers=32)

        try:
            futures = []
            for row in tqdm.tqdm(
                    raw_df.itertuples(),
                    total=len(raw_df),
                    desc="Submitting futures",
                ):
                futures.append(parallel.submit(self.annotate_one, row.Index, row))

            for future in tqdm.tqdm(
                    futures,
                    desc="Annotating",
                ):
                try:
                    result = future.result()
                except ValueError as e:
                    if e.args[0] == "UNKNOWN_ERROR":
                        # Retryable, skip for now
                        continue
                    else:
                        raise

                sv_lat[result.key] = result.lat
                sv_lng[result.key] = result.lng
                sv_status[result.key] = result.status
                sv_pano_id[result.key] = result.pano_id
        finally:
            parallel.shutdown()

        out_df['latitude'] = sv_lat
        out_df['longitude'] = sv_lng
        out_df['status'] = sv_status
        out_df['pano_id'] = sv_pano_id
        return out_df

    # Parameterize a SV dataframe
    def parameterize_random_heading(self, sv_df):
        out_df = sv_df.drop_duplicates()
        out_df = out_df.query("status == 'OK'")

        out_df['fov'] = 45
        out_df['pitch'] = 0

        # Randomize the heading
        out_df['heading'] = numpy.random.randint(0, 360, len(out_df))

        # Generate a path to save each image to
        paths = []
        for row in tqdm.tqdm(out_df.itertuples()):
            paths.append(f"sv_{row.pano_id}.jpg")
        out_df['img_path'] = paths

        # Randomize in case the input isn't
        return out_df.sample(frac=1)

    # Create a URLs and paths dataframe
    def generate_urls_paths(self, in_df):
        out_columns = {"img_path": [], "url": []}

        for row in tqdm.tqdm(
                in_df.itertuples(),
                total=len(in_df),
                desc="Generating URLs",
            ):
            url = self._sign_url(
                "https://maps.googleapis.com/maps/api/streetview"
                f"?pano={row.pano_id}"
                f"&size=640x640"
                f"&fov={row.fov}"
                f"&heading={row.heading}"
                f"&key={self.api_key}"
            )

            out_columns["img_path"].append(row.img_path)
            out_columns["url"].append(url)

        return pandas.DataFrame(out_columns)

    # Do all steps, saving intermediate results to disk
    def transform_files(self, df_dir, rows_to_annotate: Optional[int]):
        # Do all stages
        df_dir = Path(df_dir)
        raw_df = pandas.read_pickle(df_dir / "s1_raw.pkl")

        #prev_ann_df = None
        #ann_df_path = df_dir / "s2_annotated.pkl"
        #if ann_df_path.is_file():
        #    prev_ann_df = pandas.read_pickle(ann_df_path)

        #    # Exclude already processed rows
        #    unprocessed = (prev_ann_df['status'] != "OK") & (prev_ann_df['status'] != "ZERO_RESULTS")
        #    if len(unprocessed) < len(raw_df):
        #        unprocessed = pandas.concat([unprocessed, pandas.Series([True] * (len(raw_df) - len(unprocessed)))], ignore_index=True)
        #    unprocessed = unprocessed.head(len(raw_df))
        #    raw_df = raw_df.loc[unprocessed]
        #    print(f"Not yet processed: {len(raw_df)} rows")

        ## Limit the number of rows to process
        #if rows_to_process is not None:
        #    raw_df = raw_df.head(rows_to_process)
        #    print(f"Processing {len(raw_df)} rows")

        #if len(raw_df) == 0:
        #    print("Nothing to annotate!")
        #else:
        #    print("Annotating")
        #    ann_df = self.annotate(raw_df)
        #    if prev_ann_df is not None:
        #        ann_df = pandas.concat([prev_ann_df, ann_df])
        #    ann_df.to_pickle(ann_df_path)

        # TODO: how to combine this with incremental annotation?
        ann_df_path = df_dir / "s2_annotated.pkl"
        if ann_df_path.is_file():
            ann_df = pandas.read_pickle(ann_df_path)

        param_df_path = df_dir / "s3_parameterized.pkl"
        if param_df_path.is_file():
            param_df = pandas.read_pickle(param_df_path)
        else:
            print("Parameterizing")
            param_df = self.parameterize_random_heading(ann_df)
            param_df.to_pickle(param_df_path)

        urls_paths_df_path = df_dir / "_urls_paths.pkl"
        if urls_paths_df_path.is_file():
            urls_paths_df = pandas.read_pickle(urls_paths_df_path)
        else:
            print("Generating URLs")
            urls_paths_df = self.generate_urls_paths(param_df)
            urls_paths_df.to_pickle(urls_paths_df_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_path", type=str, required=True, help="Google Maps API key")
    parser.add_argument("--sign_secret_path", type=str, required=True, help="Google Maps API signing secret")
    parser.add_argument("--df_dir", type=str, required=True, help="Directory with dataframes")
    parser.add_argument("--rows_to_annotate", type=int, help="Limit the number of rows to annotate")
    args = parser.parse_args()

    with open(args.api_key_path) as f:
        api_key = f.read().strip()
    with open(args.sign_secret_path) as f:
        sign_secret = f.read().strip()

    creator = StreetViewDatasetCreator(api_key, sign_secret)

    # Do all stages
    creator.transform_files(args.df_dir, args.rows_to_annotate)

if __name__ == "__main__":
    main()