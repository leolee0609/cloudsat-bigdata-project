#!/usr/bin/env python
"""
chunk_h5_spark.py
-----------------
Use Spark to read a large HDF5 (or many HDF5) from some distributed FS,
then write out a chunked Parquet dataset. 
(This example is simplified; real-world usage often requires more advanced 
streaming or external libraries to read HDF5 in parallel.)

Usage:
  spark-submit chunk_h5_spark.py --input_h5 /path/to/huge_input.h5 --output_dir /path/to/parquet_dir
"""

import os
import sys
import argparse
import h5py
import numpy as np
from pyspark.sql import SparkSession, Row

def chunk_h5_spark(input_h5, output_dir, chunk_size=10000):
    spark = SparkSession.builder.appName("ChunkH5Spark").getOrCreate()
    sc = spark.sparkContext

    # This approach is simplistic. If you can't read the entire HDF5 in one node's memory, 
    # you might need an HPC or a specialized parallel-HDF5 library. 
    # We'll assume input_h5 is accessible from the driver node (or from each executor).
    
    with h5py.File(input_h5, 'r') as f:
        input_ds = f['input_features']
        total_samples = input_ds.shape[0]

    # We'll parallelize chunk indices, read chunk slices in each task, then produce RDD rows:
    indices = [(start, min(start+chunk_size, total_samples)) 
               for start in range(0, total_samples, chunk_size)]

    def read_chunk_in_worker(index_slice):
        import h5py
        input_h5_local = input_h5  # closure
        s, e = index_slice
        rows = []
        with h5py.File(input_h5_local, 'r') as f:
            data_chunk = f['input_features'][s:e]  # shape: (N, seq_len, features) or (N,H,C)
        # Convert the chunk to Rows. We can flatten or store as a list type. 
        # Suppose shape is (N, H, C). We'll store "t, h, channel_data"
        # So row => Row(index=int, t=int, h=int, channels=list_of_float)
        # This is an example of how you'd create row-based data for Spark:
        # In practice, you might want a more columnar approach.
        for i, arr in enumerate(data_chunk):
            global_idx = s + i
            # arr shape: (H, C)
            Hdim, Cdim = arr.shape
            for h in range(Hdim):
                pixel = arr[h].tolist()  # list of float
                # create row
                rows.append(Row(
                    index=global_idx,
                    height=h,
                    channels=pixel
                ))
        return rows

    rdd = sc.parallelize(indices, len(indices)) \
            .flatMap(read_chunk_in_worker)

    # Now we have an RDD of Row objects. Convert to DataFrame:
    df = spark.createDataFrame(rdd)
    df.write.mode('overwrite').parquet(output_dir)
    spark.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=10000)
    args = parser.parse_args()

    chunk_h5_spark(args.input_h5, args.output_dir, args.chunk_size)

if __name__ == "__main__":
    main()
