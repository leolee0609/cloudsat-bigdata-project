#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
import ast
from pyspark.sql import SparkSession

def parsePixelEventLine(line):
    """
    Each line from the BFS-labeled output looks like:
      "((file_no, t, x, y, h), event_id)"

    We'll parse that into:
      ((file_no, t, x, y, h), event_id)
    """
    return ast.literal_eval(line.strip())

def unionSets(set1, set2):
    """Return the union of two Python sets."""
    return set1.union(set2)

def main(pixel2event_path, output_path):
    spark = SparkSession.builder.appName("InvertedIndexEvents_XYTH").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")  # only show errors

    # Remove output directory if it exists (LOCAL FS)
    if os.path.exists(output_path):
        print(f"[INFO] Removing existing directory: {output_path}")
        shutil.rmtree(output_path)

    #----------------------------------------------------------------------------
    # 1) Read BFS-labeled data from pixel2event_path
    #    Each line: "((file_no, t, x, y, h), event_id)"
    #----------------------------------------------------------------------------
    labeled_rdd = spark.sparkContext.textFile(pixel2event_path).map(parsePixelEventLine)
    # => RDD[ ((file_no, t, x, y, h), event_id) ]

    #----------------------------------------------------------------------------
    # 2) Invert the mapping -> (event_id, (file_no, t, x, y, h))
    #----------------------------------------------------------------------------
    inverted_rdd = labeled_rdd.map(lambda kv: (kv[1], kv[0]))
    # => RDD[ (event_id, (file_no, t, x, y, h)) ]

    #----------------------------------------------------------------------------
    # 3) Group or reduce to collect all pixels for each event_id.
    #    We'll use reduceByKey with set-union to avoid duplicates
    #----------------------------------------------------------------------------
    inverted_as_sets = inverted_rdd.mapValues(lambda px: {px})
    # => RDD[ (event_id, set( (file_no, t, x, y, h) )) ]

    grouped = inverted_as_sets.reduceByKey(unionSets)
    # => RDD[ (event_id, setOfAllPixels) ]

    #----------------------------------------------------------------------------
    # 4) Convert final sets to lists for easy serialization
    #----------------------------------------------------------------------------
    event_to_pixels_rdd = grouped.map(lambda kv: (kv[0], list(kv[1])))
    # => RDD[ (event_id, [(file_no, t, x, y, h), ...]) ]

    #----------------------------------------------------------------------------
    # 5) Write final inverted index as text
    #    Each line: "(event_id, [(file_no, t, x, y, h), (file_no, t2, x2, y2, h2), ...])"
    #----------------------------------------------------------------------------
    event_to_pixels_rdd.saveAsTextFile(output_path)
    print(f"[INFO] Wrote inverted index to {output_path}")

    spark.stop()

if __name__ == "__main__":
    """
    Example usage:
      spark-submit spark_inverted_index_xyth.py \
        ./pixel2event_xyth \
        ./event_to_pixels_out_xyth

    This script expects BFS-labeled lines like:
        "((file_no, t, x, y, h), event_id)"
    and writes lines like:
        "(event_id, [(file_no, t, x, y, h), ...])"
    """
    parser = argparse.ArgumentParser(
        description="Build an inverted index from BFS-labeled pixel->event mappings (XYTH keys)."
    )
    parser.add_argument("pixel2event_path", help="Path to BFS-labeled directory with lines ((file_no, t, x, y, h), event_id).")
    parser.add_argument("output_path", help="Directory to write the event->pixels inverted index.")
    args, extra = parser.parse_known_args()

    main(args.pixel2event_path, args.output_path)
