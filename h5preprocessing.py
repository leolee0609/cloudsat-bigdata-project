#!/usr/bin/env python
"""
chunk_h5_spark.py
-----------------
Use Spark to read a large HDF5 (or many HDF5) from some distributed FS,
then write out a chunked Parquet dataset. 
(This example is simplified; real-world usage often requires more advanced 
streaming or external libraries to read HDF5 in parallel.)

Usage:
  spark-submit chunk_h5_spark.py --input_h5 /path/to/huge_input.h5 \
                                 --output_dir /path/to/parquet_dir
Example:
  spark-submit chunk_h5_spark.py \
    --input_h5 /data/huge_file.h5 \
    --output_dir /data/parquet_output
  # This will remove /data/parquet_output if it exists, then write a new Parquet dataset
"""

import os
import sys
import argparse
import h5py
import numpy as np
import shutil
from pyspark.sql import SparkSession, Row

def chunk_h5_spark(input_h5, output_dir, chunk_size=50000):
    """
    1) Reads the 'input_features' dataset from the HDF5 file (shape: (N, H, C) or (N, seq_len, features)).
    2) Splits it into chunk slices of size 'chunk_size'.
    3) Each slice is read in a parallel Spark task, 
       converted into multiple rows (one per height bin).
    4) Writes the final result as a Parquet dataset.

    :param input_h5: Path to the big HDF5 file
    :param output_dir: Path to write Parquet
    :param chunk_size: Number of "footprints" or top-level samples per chunk
    """
    spark = SparkSession.builder.appName("ChunkH5Spark").getOrCreate()
    sc = spark.sparkContext

    # Step 1) Determine total samples from the HDF5 'input_features' dataset
    # This approach is naive and reads from the driver. 
    # If the entire file can't fit in driver memory, consider HPC approaches or specialized libs.
    with h5py.File(input_h5, 'r') as f:
        input_ds = f['input_features']
        total_samples = input_ds.shape[0]
    print(f"Total samples: {total_samples}, chunk_size: {chunk_size}")

    # Step 2) Create chunk indices
    indices = [(start, min(start + chunk_size, total_samples))
               for start in range(0, total_samples, chunk_size)]
    print(f"Number of chunks: {len(indices)}")

    def read_chunk_in_worker(index_slice):
        """
        Worker function that:
         - opens the HDF5 file,
         - reads the chunk [s:e],
         - converts each row in that chunk to multiple "pixel" rows 
           if shape is (H, C).
        """
        import h5py
        s, e = index_slice
        rows = []

        # Reopen HDF5 in the worker
        with h5py.File(input_h5, 'r') as f_local:
            data_chunk = f_local['input_features'][s:e]  
            # shape could be (N, H, C) in typical usage

        # For demonstration: each row => (index, height, channels)
        for i, arr in enumerate(data_chunk):
            global_idx = s + i
            # If arr shape = (H, C):
            # We'll store row(index=global_idx, height=h, channels=list_of_floats)
            Hdim, Cdim = arr.shape
            for h in range(Hdim):
                channels_list = arr[h].tolist()  # convert to Python list
                rows.append(Row(
                    index=global_idx,
                    height=h,
                    channels=channels_list
                ))
        return rows

    # Step 3) Distribute the chunk reading via Spark
    rdd = sc.parallelize(indices, len(indices)).flatMap(read_chunk_in_worker)

    # Step 4) Convert to DataFrame and write to Parquet
    df = spark.createDataFrame(rdd)
    df.write.mode('overwrite').parquet(output_dir)

    spark.stop()
    print(f"Chunked data written to Parquet at: {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_h5", required=True, help="Path to the HDF5 file containing 'input_features'")
    parser.add_argument("--output_dir", required=True, help="Directory to write the Parquet files")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of top-level samples per chunk")
    args = parser.parse_args()

    # Remove existing output directory if present
    if os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir, ignore_errors=True)

    chunk_h5_spark(args.input_h5, args.output_dir, args.chunk_size)

if __name__ == "__main__":
    main()
