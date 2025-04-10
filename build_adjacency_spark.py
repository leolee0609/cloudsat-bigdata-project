#!/usr/bin/env python
"""
build_adjacency_spark.py
------------------------
Reads chunked data (Parquet or .h5) in Spark, builds adjacency for 
(event) pixels in a distributed manner, and writes adjacency as 
partitioned JSON lines.

This script removes any existing files in the output directory
before it saves new output.

Usage:
  spark-submit build_adjacency_spark.py \
    --input_dir /path/to/parquet_dir \
    --output_dir /path/to/adjacency_dir \
    [--file_id mydata]

Example Steps:
1. spark-submit chunk_h5_spark.py ...  # produce a Parquet of (index, height, channels)
2. spark-submit build_adjacency_spark.py --input_dir parquet_dir --output_dir adjacency_dir
"""

import os
import sys
import argparse
import json
import shutil

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

EVENT_CHANNEL_INDEX = 0  # which channel to check for > 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Parquet or chunked dir")
    parser.add_argument("--output_dir", required=True, help="Output adjacency dir (JSON)")
    parser.add_argument("--file_id", default="some_id", help="ID prefix or partition identifier for each pixel")
    args = parser.parse_args()

    # ---------------------------------------------------
    # 1) Remove existing output directory if it exists
    # ---------------------------------------------------
    if os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir, ignore_errors=True)

    # ---------------------------------------------------
    # 2) Create Spark session & read input
    # ---------------------------------------------------
    spark = SparkSession.builder.appName("BuildAdjacency").getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "200")  # or any suitable number

    # For demonstration, assume the input is Parquet with columns: index, height, channels
    df = spark.read.parquet(args.input_dir)

    # ---------------------------------------------------
    # 3) Filter event pixels: channels[0] > 4.0 
    #    (adjust threshold or channel index as needed)
    # ---------------------------------------------------
    df_events = df.filter(F.col("channels")[EVENT_CHANNEL_INDEX] > 4.0)

    # Add a "pixel_id" struct containing (file_id, t, h)
    df_events = df_events.withColumn(
        "pixel_id",
        F.struct(
            F.lit(args.file_id).alias("file_id"),
            F.col("index").alias("t"),
            F.col("height").alias("h")
        )
    )

    # ---------------------------------------------------
    # 4) Convert to an RDD of pixel_id => e.g. 
    #    Row(file_id=..., t=..., h=...)
    # ---------------------------------------------------
    rdd_events = df_events.select("pixel_id").rdd.map(lambda r: r["pixel_id"])

    # ---------------------------------------------------
    # 5) Build adjacency in each partition
    #    (We store pixel -> True in a dictionary, then 
    #     for each pixel, look up neighbors.)
    # ---------------------------------------------------
    def map_pixels(iterator):
        import collections
        pixmap = collections.defaultdict(lambda: False)
        pixels = list(iterator)

        # Mark each pixel in a local dict
        for p in pixels:
            pixmap[(p["file_id"], p["t"], p["h"])] = True

        # Generate adjacency
        results = []
        for p in pixels:
            file_id, t, h = p["file_id"], p["t"], p["h"]
            neighbors = []
            # up
            if pixmap.get((file_id, t - 1, h), False):
                neighbors.append([file_id, t - 1, h])
            # down
            if pixmap.get((file_id, t + 1, h), False):
                neighbors.append([file_id, t + 1, h])
            # left
            if pixmap.get((file_id, t, h - 1), False):
                neighbors.append([file_id, t, h - 1])
            # right
            if pixmap.get((file_id, t, h + 1), False):
                neighbors.append([file_id, t, h + 1])

            # key:   (file_id, t, h)
            # value: list of neighbors
            results.append(((file_id, t, h), neighbors))

        return iter(results)

    adjacency_rdd = rdd_events.mapPartitions(map_pixels)
    # adjacency_rdd => ( (file_id, t, h), [[file_id, t_n, h_n], ... ] )

    # ---------------------------------------------------
    # 6) Convert adjacency to JSON lines & save
    # ---------------------------------------------------
    def to_json_lines(x):
        key, nbrs = x
        return json.dumps({"key": list(key), "value": nbrs})

    adjacency_rdd.map(to_json_lines).saveAsTextFile(args.output_dir)

    spark.stop()
    print("Adjacency build complete. Written to:", args.output_dir)

if __name__ == "__main__":
    main()
