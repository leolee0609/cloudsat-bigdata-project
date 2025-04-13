#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
import ast
from pyspark.sql import SparkSession

def expandOneStep(record):
    """
    Given (p, neighborsSet),
    emit (p, neighborsSet) so p retains its old info,
    AND for each neighbor q in neighborsSet, emit (q, neighborsSet)
    so q learns about p's connectivity.
    """
    p, neighbors = record
    # Emit the original pair
    yield (p, neighbors)
    # Emit for each neighbor
    for q in neighbors:
        yield (q, neighbors)

def unionSets(set1, set2):
    """Return the union of two Python sets."""
    return set1.union(set2)

def parseAdjacencyLine(line):
    """
    Parse a line of the form:
       "((file_no, t, x, y, h), [(file_no, t2, x2, y2, h2), ...])"
    into:
       ((file_no, t, x, y, h), set_of_neighbors)

    Where each neighbor is also (file_no, tNeighbor, xNeighbor, yNeighbor, hNeighbor).
    """
    kv = ast.literal_eval(line.strip())      
    # e.g. kv => ((file_no, t, x, y, h), [(file_no, t2, x2, y2, h2), (file_no, t3, x3, y3, h3), ...])
    key, neighbor_list = kv
    neighbor_set = set(neighbor_list)       
    return (key, neighbor_set)

def main(adjacency_input, output_path):
    """
    Parallel BFS for connected components, where the adjacency lines are keyed by
    (file_no, t, x, y, h).

    1) Reads adjacency lines of the form:
         "((file_no, t, x, y, h), [(file_no, t2, x2, y2, h2), ...])"
    2) Iteratively merges adjacency sets until no new neighbors appear (BFS).
    3) Assigns each connected component a unique event_id.
    4) Outputs lines: "(((file_no, t, x, y, h), event_id))"
    """
    spark = SparkSession.builder.appName("ParallelBFSConnectedComponents_XYTH").getOrCreate()

    # Hide Spark's own INFO logs, show only errors
    spark.sparkContext.setLogLevel("ERROR")

    # Remove output directory if it exists
    if os.path.exists(output_path):
        print(f"[INFO] Removing existing directory: {output_path}")
        shutil.rmtree(output_path)

    # 1) Read adjacency from textFile
    adjacency_rdd = spark.sparkContext.textFile(adjacency_input).map(parseAdjacencyLine)
    # => RDD[((file_no, t, x, y, h), set_of_neighbors)]

    # 2) Initialize BFS state = immediate adjacency
    bfs_state_rdd = adjacency_rdd
    iteration = 0

    # 3) Iteratively expand adjacency until no new neighbors
    while True:
        iteration += 1
        # Expand adjacency
        expanded = bfs_state_rdd.flatMap(expandOneStep)
        # => for each (p, set_of_neighbors) => (p, set_of_neighbors) + (neighbor, set_of_neighbors)

        # Merge adjacency
        new_bfs_state_rdd = expanded.reduceByKey(unionSets)

        # Check BFS delta (# of newly discovered neighbors this iteration)
        joined = bfs_state_rdd.join(new_bfs_state_rdd)  
        # => (p, (oldSet, newSet))

        bfs_delta = joined.mapValues(lambda pair: len(pair[1] - pair[0])) \
                          .values() \
                          .sum()

        print(f"Iteration {iteration}, BFS delta = {bfs_delta}")

        if bfs_delta == 0:
            # Converged
            break
        else:
            bfs_state_rdd = new_bfs_state_rdd

    # 4) Assign each connected component a unique event_id
    # Ensure each pixel p is in its adjacency set
    final_bfs_rdd = bfs_state_rdd.map(lambda kv: (kv[0], kv[1].union({kv[0]})))
    # => (p, neighborsSet ∪ {p})

    # Representative = min(all connected pixels in that set)
    pixel_representative_rdd = final_bfs_rdd.mapValues(lambda s: min(s))
    # => (p, representative)

    # Distinct reps => (rep, event_id) pairs
    distinct_reps = pixel_representative_rdd.values().distinct()
    reps_with_id = distinct_reps.zipWithIndex()
    # => (representative, event_id)

    # Now join each p with its representative's event_id
    # pixel_representative_rdd => (p, rep)
    # reps_with_id => (rep, event_id)
    joined_rdd = pixel_representative_rdd.map(lambda kv: (kv[1], kv[0])) \
                                         .join(reps_with_id)
    # => (rep, (p, event_id))

    # final: (p, event_id)
    final_labeled_rdd = joined_rdd.map(lambda kv: (kv[1][0], kv[1][1]))

    # 5) Save final results as text
    final_labeled_rdd.saveAsTextFile(output_path)
    print(f"[INFO] Wrote BFS-labeled connected components to {output_path}")

    spark.stop()

if __name__ == "__main__":
    """
    Usage:
      spark-submit bfs_component_label_xyth.py \
          /path/to/adjacency_out_xyth \
          /path/to/bfs_labeled_out

    Where adjacency_out_xyth has lines like:
      "((chunk_5000_10000.h5, 769287740.0, 45.1234, -120.5678, 30), 
        [(chunk_5000_10000.h5, 769287740.0, 45.1234, -120.5678, 31), ...])"

    This BFS script merges adjacency sets until converged, 
    and then assigns each connected component an event_id.
    """
    parser = argparse.ArgumentParser(
        description="Parallel BFS for XYTH keys: (file_no, t, x, y, h)."
    )
    parser.add_argument("adjacency_input", help="Path to adjacency text files keyed by (file_no, t, x, y, h).")
    parser.add_argument("output_path", help="Directory to write BFS-labeled connected components.")
    args, extra = parser.parse_known_args()

    main(args.adjacency_input, args.output_path)
