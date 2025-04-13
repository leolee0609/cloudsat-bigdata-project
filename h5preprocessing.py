#!/usr/bin/env python3

import os
import shutil
import h5py
import argparse
import numpy as np
from tqdm import tqdm

def chunkify_hdf5(input_file_path, output_dir, chunk_size=10000):
    """
    Reads the 'input_features' and 'footprint_attributes' datasets from a large HDF5 file,
    splits them into smaller chunks, and saves each chunk as a separate HDF5 file.

    :param input_file_path: Path to the large HDF5 file.
    :param output_dir: Directory where chunk files will be saved.
    :param chunk_size: Number of footprints per chunk.
    """
    # If the output directory exists, remove it entirely
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Open the large HDF5 file in read mode
    with h5py.File(input_file_path, 'r') as f:
        input_features = f['input_features']          # shape = (N, height, channels)
        footprint_attributes = f['footprint_attributes']  # shape = (N, num_attrs)

        num_footprints = input_features.shape[0]
        print(f"Total number of footprints: {num_footprints}")

        # Iterate through the footprints in increments of chunk_size
        for start_idx in tqdm(range(0, num_footprints, chunk_size), desc="Chunking data"):
            end_idx = min(start_idx + chunk_size, num_footprints)

            # Read the chunk slice
            chunk_input_features = input_features[start_idx:end_idx]
            chunk_footprint_attrs = footprint_attributes[start_idx:end_idx]

            # Construct chunk filename (e.g., chunk_0_9999.h5)
            chunk_filename = f"chunk_{start_idx}_{end_idx}.h5"
            chunk_path = os.path.join(output_dir, chunk_filename)

            # Write chunk to a new HDF5 file
            with h5py.File(chunk_path, 'w') as chunk_f:
                chunk_f.create_dataset(
                    'input_features',
                    data=chunk_input_features,
                    compression='gzip',       # optional compression
                    compression_opts=4
                )
                chunk_f.create_dataset(
                    'footprint_attributes',
                    data=chunk_footprint_attrs,
                    compression='gzip',
                    compression_opts=4
                )

def main():
    parser = argparse.ArgumentParser(description="Chunk a large HDF5 dataset into smaller files.")
    parser.add_argument("--input_file", required=True, help="Path to the large HDF5 file.")
    parser.add_argument("--output_dir", required=True, help="Directory where chunk files will be saved.")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Number of footprints per chunk (default: 10000).")

    args = parser.parse_args()

    chunkify_hdf5(args.input_file, args.output_dir, args.chunk_size)

if __name__ == "__main__":
    main()
