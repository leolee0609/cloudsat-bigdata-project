#!/usr/bin/env python
"""
build_adjacency_spark.py
------------------------
Read chunked data (Parquet or .h5) in Spark, build adjacency for 
(event) pixels in a distributed manner, write adjacency as 
partitioned Parquet or JSON.

Usage:
  spark-submit build_adjacency_spark.py \
    --input_dir /path/to/parquet_dir \
    --output_dir /path/to/adjacency_dir
"""

import os
import sys
import argparse
import json
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

EVENT_CHANNEL_INDEX = 0  # which channel to check for > 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Parquet or chunked dir")
    parser.add_argument("--output_dir", required=True, help="Output adjacency dir")
    parser.add_argument("--file_id", default="some_id", help="If needed, or partition by file_id")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("BuildAdjacency").getOrCreate()

    # Read the chunked data (assuming we used the example chunk_h5_spark => columns: index, height, channels)
    df = spark.read.parquet(args.input_dir)

    # We want to find event pixels.  channels[0] > 0
    # In Spark SQL, we can do something like:
    df_events = df.filter(F.col("channels")[EVENT_CHANNEL_INDEX] > 0)

    # Now we need to find neighbors. 
    # Because each row is (index, height, channels), let's define a "pixel_id" = (file_id, index, height).
    # We'll keep it as a string or a struct. We'll store file_id from user param or maybe from partition info.
    df_events = df_events.withColumn("pixel_id", F.struct(
        F.lit(args.file_id).alias("file_id"),
        F.col("index").alias("t"),
        F.col("height").alias("h")
    ))

    # Next: We must identify neighbors. Typically, we'd need to 
    # self-join on conditions: |t1 - t2| + |h1 - h2| = 1. 
    # But that can be large. 
    # We can do window-based or create an RDD approach. For illustration, let's do an RDD approach:

    rdd_events = df_events.select("pixel_id").rdd.map(lambda r: r["pixel_id"])
    # => RDD of Row(file_id=..., t=..., h=...)

    # We'll transform this into a dictionary pixel -> True, then for each pixel, we check neighbors:
    # This approach is naive but demonstrates the concept.

    def map_pixels(iterator):
        import collections
        pixmap = collections.defaultdict(lambda: False)
        # load all pixels in memory if possible
        pixels = list(iterator)
        for p in pixels:
            pixmap[(p["file_id"], p["t"], p["h"])] = True

        # Now generate adjacency
        results = []
        for p in pixels:
            file_id, t, h = p["file_id"], p["t"], p["h"]
            neighbors = []
            # up
            if pixmap.get((file_id, t-1, h), False):
                neighbors.append([file_id, t-1, h])
            # down
            if pixmap.get((file_id, t+1, h), False):
                neighbors.append([file_id, t+1, h])
            # left
            if pixmap.get((file_id, t, h-1), False):
                neighbors.append([file_id, t, h-1])
            # right
            if pixmap.get((file_id, t, h+1), False):
                neighbors.append([file_id, t, h+1])
            results.append(((file_id, t, h), neighbors))
        return iter(results)

    adjacency_rdd = rdd_events.mapPartitions(map_pixels)

    # adjacency_rdd => ( (file_id, t, h), [[file_id, t', h'], ...] )

    # Convert to DataFrame or just write as JSON lines
    def to_json_lines(x):
        key, nbrs = x
        return json.dumps({
            "key": list(key),
            "value": nbrs
        })

    adjacency_rdd.map(to_json_lines).saveAsTextFile(args.output_dir)

    spark.stop()

if __name__ == "__main__":
    main()
