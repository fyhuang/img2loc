"""Read ways from OSM extract, along with node locations, and write to Parquet."""

import argparse
import json
from pathlib import Path

import osmium
import tqdm
import pandas as pd

import pyarrow as pa
from pyarrow import parquet as pq

PLANET_EXTRACT = Path.home() / "datasets" / "osm" / "planet-240513.osm.pbf"
ANTARCTICA_EXTRACT = Path.home() / "datasets" / "osm" / "antarctica-latest.osm.pbf"

SCHEMA = \
    pa.schema([
        ("way_id", pa.int64()),
        ("tag_key", pa.string()),
        ("tag_value", pa.string()),
        ("waypoint_num", pa.int64()),
        ("lat", pa.float64()),
        ("lng", pa.float64()),
    ])

class WayExpansionHandler(osmium.SimpleHandler):
    def __init__(self, output_func):
        super().__init__()
        self.output_func = output_func

    def way(self, w):
        if w.deleted:
            return
        if not w.visible:
            return

        try:
            point_list = [(n.lat, n.lon) for n in w.nodes]
        except osmium.InvalidLocationError:
            # Possibly a way where nodes are not in the extract
            return

        self.output_func(w.id, w.tags, point_list)




class WayParquetWriter:
    def __init__(self, out_path, progresshook):
        self._writer = pq.ParquetWriter(out_path, SCHEMA)
        self._progresshook = progresshook

        self._init_batch()

    def _init_batch(self):
        self.processed = 0
        self.batch_columns = {
            "way_id": [],
            # tags
            "tag_key": [],
            "tag_value": [],
            # points
            "waypoint_num": [],
            "lat": [],
            "lng": [],
        }

    def _write_batch(self):
        # Convert and write batch
        #df = pd.DataFrame(
        #    {
        #        "way_id": self.batch_columns["way_id"],
        #        "tag_key": pd.Series(self.batch_columns["tag_key"], dtype=pd.StringDtype())),
        #        "tag_value": pd.Series(self.batch_columns["tag_value"], dtype=pd.StringDtype())),
        #    }
        #)
        rb = pa.RecordBatch.from_pydict(
            self.batch_columns,
            schema=SCHEMA,
        )
        self._writer.write_batch(rb)
        self._progresshook(1)

        self._init_batch()

    def _maybe_write_batch(self):
        #if len(self.batch_columns["way_id"]) >= 2048:
        if self.processed >= 2048:
            self._write_batch()

    def write_way(self, way_id, tags, point_list):
        # Write tags
        for t in tags:
            self.batch_columns["way_id"].append(way_id)
            self.batch_columns["tag_key"].append(t.k)
            self.batch_columns["tag_value"].append(t.v)
            self.batch_columns["waypoint_num"].append(None)
            self.batch_columns["lat"].append(None)
            self.batch_columns["lng"].append(None)

        # Write points
        for i,p in enumerate(point_list):
            self.batch_columns["way_id"].append(way_id)
            self.batch_columns["tag_key"].append(None)
            self.batch_columns["tag_value"].append(None)
            self.batch_columns["waypoint_num"].append(i)
            self.batch_columns["lat"].append(p[0])
            self.batch_columns["lng"].append(p[1])

        self.processed += 1

        self._maybe_write_batch()

    def finish(self):
        if len(self.batch_columns["way_id"]) > 0:
            self._write_batch()
        self._writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    with tqdm.tqdm() as pb:
        pwriter = WayParquetWriter(args.output, pb.update)
        handler = WayExpansionHandler(pwriter.write_way)
        #handler.apply_file(PLANET_EXTRACT, locations=True, idx='dense_file_array,planet.nodecache')
        handler.apply_file(ANTARCTICA_EXTRACT, locations=True, idx='sparse_mem_array')
        pwriter.finish()

if __name__ == "__main__":
    main()