#!/usr/bin/env python
"""
invert_bfs_labels.py
--------------------
Build an inverted index from (pixel -> label) to (label -> [pixels])
using PySpark. Typically used after BFS/connected-component labeling.

Usage:
  spark-submit invert_bfs_labels.py \
    --bfs_output_dir /path/to/bfs_labeled_json \
    --inverted_index_dir /path/to/inverted_index_output

Example BFS Output JSON line:
  {"key": ["file_1", 10, 2], "value": 12}

This script will produce lines like:
  {"label": 12, "pixels": [["file_1", 10, 2], ["file_1", 9, 2], ...]}
"""

import os
import sys
import argparse
import json
import shutil
from pyspark.sql import SparkSession

def main():
    parser = argparse.ArgumentParser(description="Invert BFS-labeled data to get event -> pixels mapping.")
    parser.add_argument("--bfs_output_dir", required=True,
                        help="Directory containing BFS-labeled JSON lines (pixel -> label).")
    parser.add_argument("--inverted_index_dir", required=True,
                        help="Output directory for the inverted index (label -> list_of_pixels).")
    args = parser.parse_args()

    # 1) Remove existing output directory if it exists
    if os.path.exists(args.inverted_index_dir):
        print(f"Removing existing directory: {args.inverted_index_dir}")
        shutil.rmtree(args.inverted_index_dir, ignore_errors=True)

    # 2) Create Spark session
    spark = SparkSession.builder.appName("InvertBFSLabels").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # 3) Read BFS-labeled data from JSON lines
    #    Format: {"key": ["file_1", 10, 2], "value": 12}
    #    => ( pixel_tuple, label_int )
    labeled_rdd = sc.textFile(args.bfs_output_dir) \
                    .map(json.loads) \
                    .map(lambda record: (tuple(record["key"]), record["value"]))

    if labeled_rdd.isEmpty():
        print("No BFS-labeled data found. Exiting.")
        spark.stop()
        return

    # 4) Invert each record => (label, pixel)
    label_pixel_rdd = labeled_rdd.map(lambda x: (x[1], x[0]))

    # 5) Group all pixels by label => (label -> [pixel1, pixel2, ...])
    #    This is the 'reduceByKey' or 'groupByKey' step in a MapReduce sense.
    event_to_pixels_rdd = label_pixel_rdd.groupByKey().mapValues(list)

    # 6) Convert each (label, [pixels]) to JSON lines
    def to_json_line(item):
        label, pix_list = item
        # pix_list is a list of tuples => convert each to a list for JSON
        pix_list_json = [list(pix) for pix in pix_list]
        return json.dumps({
            "label": label,
            "pixels": pix_list_json
        })

    inverted_rdd = event_to_pixels_rdd.map(to_json_line)

    # 7) Save to partitioned JSON lines
    inverted_rdd.saveAsTextFile(args.inverted_index_dir)

    print(f"Inverted index saved to {args.inverted_index_dir}")
    spark.stop()

if __name__ == "__main__":
    main()
