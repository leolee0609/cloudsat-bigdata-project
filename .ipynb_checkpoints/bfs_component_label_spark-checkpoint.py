#!/usr/bin/env python
"""
bfs_component_label_spark.py
----------------------------
Use PySpark to run a BFS-based connected-component labeling 
on an adjacency list of event pixels.

Usage:
  spark-submit bfs_component_label_spark.py \
    --adjacency_dir /path/to/adjacency_dir \
    --output_dir /path/to/bfs_output \
    [--reducers 4]

Example adjacency format (JSON lines):
  {
    "key": ["file_1", 10, 2],
    "value": [["file_1", 9, 2], ["file_1", 11, 2]]
  }
  which means pixel ("file_1",10,2) has neighbors [("file_1",9,2), ("file_1",11,2)].
"""

import os
import sys
import argparse
import json
from pyspark.sql import SparkSession

MAX_ITERATIONS = 100

def map_phase_func(record):
    """
    BFS Map step:
    Input 'record': (pixel, (clusterSet, neighbors))
      - pixel: a tuple like ("file_1", 10, 2)
      - clusterSet: a Python set, e.g. {("file_1", 10, 2), ...}
      - neighbors: a list of neighbor pixels

    We emit:
      (pixel -> clusterSet)   # keep the current cluster info for this pixel
      (nbr   -> clusterSet)   # for each neighbor, propagate this clusterSet
    """
    pixel, (cluster_set, neighbors) = record
    out = []
    # Keep pixel => cluster_set
    out.append((pixel, cluster_set))

    # For each neighbor, pass the same cluster_set
    for n in neighbors:
        out.append((n, cluster_set))

    return out

def reduce_phase_func(a, b):
    """
    BFS Reduce step: union the cluster sets for this pixel.
    a, b are both Python sets of pixels.
    """
    return a.union(b)

def compute_delta(old_rdd, new_rdd):
    """
    Compare old_rdd vs new_rdd to see how many new members were added.
    Both are (pixel, set_of_pixels). We leftOuterJoin to see differences.
    """
    joined = old_rdd.leftOuterJoin(new_rdd)
    def count_added(row):
        pixel, (oldset, newset) = row
        oldset = oldset if oldset else set()
        added = newset.difference(oldset)
        return len(added)
    return joined.map(count_added).sum()

def assign_labels(final_rdd):
    """
    After BFS converges, final_rdd => (pixel, set_of_pixels).
    1) Convert each set_of_pixels into a sorted tuple => identify distinct connected components.
    2) zipWithUniqueId => assign each distinct set an integer label.
    3) Map each pixel's set -> label.
    """
    # Distinct sets
    cluster_sets = final_rdd.map(lambda x: tuple(sorted(x[1]))).distinct()

    # Assign a unique ID to each set; shift ID to start at 1
    set_with_id = cluster_sets.zipWithUniqueId().map(lambda x: (x[0], x[1] + 1))

    # Collect mapping into driver memory. For extremely large data, you may need a different approach.
    set_map = set_with_id.collectAsMap()
    bc_map = final_rdd.context.broadcast(set_map)

    # Map each pixel->(the ID of its cluster set)
    labeled_rdd = final_rdd.mapValues(lambda s: bc_map.value[tuple(sorted(s))])
    return labeled_rdd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjacency_dir", required=True, 
                        help="Directory of adjacency JSON lines or parquet")
    parser.add_argument("--output_dir", required=True, 
                        help="Output directory for BFS iteration + final labels")
    parser.add_argument("--reducers", type=int, default=4, 
                        help="Number of shuffle partitions (reducers)")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("SparkBFSComponentLabel").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    spark.conf.set("spark.sql.shuffle.partitions", str(args.reducers))

    # 1) Load adjacency from JSON lines
    # Each record => { "key": [file_id, t, h], "value": [[file_id, tn, hn], ...] }
    # We'll parse them into (pixel -> [neighbors]) pairs.
    adjacency_rdd = sc.textFile(args.adjacency_dir) \
                      .map(json.loads) \
                      .map(lambda d: (tuple(d["key"]), [tuple(n) for n in d["value"]]))

    if adjacency_rdd.isEmpty():
        print("No adjacency data found. Exiting.")
        spark.stop()
        return

    # 2) Initialize BFS => (pixel -> {pixel})
    # That is, each pixel starts in its own cluster set
    pixel_to_set_rdd = adjacency_rdd.map(lambda x: (x[0], {x[0]}))

    # 3) BFS iteration
    for i in range(MAX_ITERATIONS):
        # Join cluster sets with adjacency => (pixel, (clusterSet, neighbors))
        joined_rdd = pixel_to_set_rdd.join(adjacency_rdd, numPartitions=args.reducers)
        
        # Map: propagate cluster sets to neighbors
        mapped = joined_rdd.flatMap(map_phase_func)
        
        # Reduce: union sets for each pixel
        new_pixel_to_set_rdd = mapped.reduceByKey(reduce_phase_func, numPartitions=args.reducers)
        
        # Check convergence (how many new elements got added)
        changes = compute_delta(pixel_to_set_rdd, new_pixel_to_set_rdd)
        print(f"Iteration {i}, BFS delta = {int(changes)}")

        pixel_to_set_rdd = new_pixel_to_set_rdd

        if changes == 0:
            print(f"BFS converged at iteration {i}")
            break

    # 4) Assign final numeric labels per connected component
    labeled_rdd = assign_labels(pixel_to_set_rdd)

    # 5) Save final pixel->label mapping as partitioned JSON lines
    def to_json_line(x):
        (pixel, label) = x
        return json.dumps({"key": list(pixel), "value": label})

    labeled_rdd.map(to_json_line).saveAsTextFile(
        os.path.join(args.output_dir, "final_labels")
    )

    spark.stop()

if __name__ == "__main__":
    main()
