#!/usr/bin/env python
"""
chunk_h5_spark.py
-----------------
Use Spark to read a large HDF5 from some distributed FS,
then write out:
  1) A chunked Parquet dataset of the main 3D profile (N,H,C).
  2) A single Parquet file of the 2D footprint-attribute table (N, ...).

Usage:
  spark-submit chunk_h5_spark.py \
    --input_h5 /path/to/huge_input.h5 \
    --output_dir /path/to/parquet_dir \
    --chunk_size 5000
"""

import os
import sys
import argparse
import h5py
import numpy as np
import shutil
from pyspark.sql import SparkSession, Row

def chunk_h5_spark(input_h5, output_dir, chunk_size=10000):
    spark = SparkSession.builder.appName("ChunkH5Spark").getOrCreate()
    sc = spark.sparkContext

    # Remove old output if it exists
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    # Create the output directory again
    os.makedirs(output_dir, exist_ok=True)

    # 1) Read main dataset shape
    with h5py.File(input_h5, 'r') as f:
        # The main 3D dataset (N, H, C)
        if 'input_features' not in f:
            raise ValueError("HDF5 file missing 'input_features' dataset!")
        input_ds = f['input_features']
        total_samples = input_ds.shape[0]
        # The 2D footprint attributes (N, ?)
        if 'footprint_attributes' not in f:
            raise ValueError("HDF5 file missing 'footprint_attributes' dataset!")
        attr_ds = f['footprint_attributes']
        attr_shape = attr_ds.shape
        print("Footprint attributes shape:", attr_shape)

    print(f"Total footprints for input_features: {total_samples}, chunk_size: {chunk_size}")
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    print(f"Number of chunks: {num_chunks}")

    # 2) Create chunk indices
    indices = [(start, min(start + chunk_size, total_samples))
               for start in range(0, total_samples, chunk_size)]

    # 3) Worker function to read slices of the 3D data
    def read_chunk_in_worker(index_slice):
        import h5py
        s, e = index_slice
        rows = []

        with h5py.File(input_h5, 'r') as f_local:
            data_chunk = f_local['input_features'][s:e]  # shape => (size, H, C)

        # We'll assign a "file_id" or chunk_id = chunk_{chunk_idx}
        # but for now we rely on partition index. We'll store that as well.
        # We'll store each row => Row(file_no=chunk_idx, t=global_idx, height=h, channels= list_of_float)
        # We'll figure out chunk_idx in the driver after this map? We'll keep it simpler:
        # We'll rely on the global_idx approach, but BFS will store "file_id" as chunk_i. See below.

        # However, to keep a "file_id" consistent, we can embed it in the RDD
        # We'll do that outside the function or pass it in as a closure.

        # Convert each footprint to (t, h, channels)
        # shape: (Nslice, H, C)
        for i, arr in enumerate(data_chunk):
            global_idx = s + i
            Hdim, Cdim = arr.shape
            for h in range(Hdim):
                chlist = arr[h].tolist()
                rows.append(Row(
                    file_id=-1,  # temporary placeholder, we'll set properly in the driver if needed
                    t=global_idx,
                    h=h,
                    channels=chlist
                ))
        return rows

    # 4) Parallelize chunk reading
    rdd = sc.parallelize(indices, len(indices)).flatMap(read_chunk_in_worker)

    # 5) Convert to DataFrame. We store file_id=-1 here but we can set it if needed.
    # Usually we rely on "t" for identification. We'll rely on BFS code to handle chunk ID if needed.
    df = spark.createDataFrame(rdd)

    # Write the chunked profile data as Parquet
    profiles_parquet = os.path.join(output_dir, "profile_data.parquet")
    df.write.mode('overwrite').parquet(profiles_parquet)
    print(f"Chunked 3D profile data saved to {profiles_parquet}")

    # 6) Also save footprint_attributes as a single Parquet file
    # We'll read them once on the driver, then create a DF
    with h5py.File(input_h5, 'r') as f:
        attr_data = f['footprint_attributes'][:]  # shape (N, num_attrs)
    # Suppose each row => [Lat, Lon, TAI_start, Profile_time, ...]
    # We'll define columns here. Adjust as needed.
    # We'll also store 't' as the row index, so the BFS code can do time lookups.
    # Example columns:
    # 0 => Lat, 1 => Lon, 2 => TAI_start, 3 => Profile_time, ...
    # Adjust if your dataset differs
    # We don't know the exact number of columns, so let's do something dynamic
    num_attrs = attr_data.shape[1]
    # We'll guess some columns or create generic col_0 ... col_N
    # But you specifically mention TAI_start, Profile_time, etc. Let's do an example:
    # If you have EXACT columns, please adapt. We'll do a placeholder approach:
    # e.g. columns = ['Latitude','Longitude','TAI_start','Profile_time', ...]
    # For a real scenario, you'd get them from the code or a separate array
    col_names = [f"attr_{i}" for i in range(num_attrs)]
    # We'll treat row index as 't'
    # Build list of Rows
    rows_attr = []
    for i in range(attr_data.shape[0]):
        row_vals = attr_data[i].tolist()
        # Build dict
        row_dict = {"t": i}
        for cidx, val in enumerate(row_vals):
            row_dict[col_names[cidx]] = val
        rows_attr.append(Row(**row_dict))

    attr_rdd = sc.parallelize(rows_attr)
    attr_df = spark.createDataFrame(attr_rdd)
    footprint_parquet = os.path.join(output_dir, "footprint_attributes.parquet")
    attr_df.write.mode('overwrite').parquet(footprint_parquet)
    print(f"Footprint attributes saved to {footprint_parquet}")

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
