#!/usr/bin/env python3

import os
import sys
import h5py
import shutil
from pyspark.sql import SparkSession

def isEvent(pixel):
    """
    Return True if the first channel of the pixel is > 4.0.
    pixel is assumed to be a 1D array/list of channel values.
    """
    return pixel[0] > 4.0

def build_event_adjacency(file_path):
    """
    Opens an HDF5 chunk file and, for each pixel that is an event, gathers
    up/down/left/right neighbors if they are also event pixels.

    Specifically:
      - Up/Down neighbors: same footprint (i), same lat/lon/t, adjacent bin h+1 or h-1
      - Left/Right neighbors: adjacent footprint (i-1 or i+1), same bin h,
        BUT only if |t_neighbor - t_current| <= 500 (seconds).

    We also incorporate the footprint's Latitude (x) and Longitude (y).
    So the key is:   (base_name, t, lat, lon, h)
    And the adjacency list is a list of the same structure:
        [(base_name, tNeighbor, latNeighbor, lonNeighbor, hNeighbor), ...]

    We'll only include neighbors that also pass isEvent(...) and meet the time constraint.
    """
    import h5py  # ensure it's available on the worker nodes
    base_name = os.path.basename(file_path)
    results = []

    with h5py.File(file_path, 'r') as f:
        input_features = f['input_features']          # shape: (N, height, channels)
        footprint_attrs = f['footprint_attributes']   # shape: (N, num_attrs)

        num_footprints = input_features.shape[0]      # N footprints
        height_dim = input_features.shape[1]          # vertical bins

        # footprint_attrs columns, as typically laid out:
        #   0 = Latitude
        #   1 = Longitude
        #   2 = TAI_start
        #   3 = Profile_time
        idx_lat = 0
        idx_lon = 1
        idx_tai_start = 2
        idx_profile_time = 3

        # Load entire arrays into memory for convenience
        input_features_np = input_features[()]        
        footprint_attrs_np = footprint_attrs[()]

        for i in range(num_footprints):
            lat = footprint_attrs_np[i, idx_lat]
            lon = footprint_attrs_np[i, idx_lon]
            tai_start = footprint_attrs_np[i, idx_tai_start]
            prof_time = footprint_attrs_np[i, idx_profile_time]
            t_current = float(tai_start + prof_time)

            for h in range(height_dim):
                pixel = input_features_np[i, h, :]
                if isEvent(pixel):
                    # Our key: (chunk_file, t_current, lat, lon, h)
                    key = (base_name, t_current, float(lat), float(lon), h)

                    # Build adjacency for neighbors
                    neighbors = []

                    # Up neighbor => same footprint i, bin h+1
                    if h + 1 < height_dim:
                        nb_pixel = input_features_np[i, h + 1, :]
                        if isEvent(nb_pixel):
                            # same lat/lon/t
                            neighbors.append((base_name, t_current, float(lat), float(lon), h + 1))

                    # Down neighbor => same footprint i, bin h-1
                    if h - 1 >= 0:
                        nb_pixel = input_features_np[i, h - 1, :]
                        if isEvent(nb_pixel):
                            neighbors.append((base_name, t_current, float(lat), float(lon), h - 1))

                    # Left neighbor => footprint i-1, same h
                    if i - 1 >= 0:
                        lat_left = footprint_attrs_np[i - 1, idx_lat]
                        lon_left = footprint_attrs_np[i - 1, idx_lon]
                        tai_left = footprint_attrs_np[i - 1, idx_tai_start]
                        prof_left = footprint_attrs_np[i - 1, idx_profile_time]
                        t_left = float(tai_left + prof_left)
                        nb_pixel = input_features_np[i - 1, h, :]
                        # check time constraint + isEvent
                        if isEvent(nb_pixel) and abs(t_left - t_current) <= 500.0:
                            neighbors.append((base_name, t_left, float(lat_left), float(lon_left), h))

                    # Right neighbor => footprint i+1, same h
                    if i + 1 < num_footprints:
                        lat_right = footprint_attrs_np[i + 1, idx_lat]
                        lon_right = footprint_attrs_np[i + 1, idx_lon]
                        tai_right = footprint_attrs_np[i + 1, idx_tai_start]
                        prof_right = footprint_attrs_np[i + 1, idx_profile_time]
                        t_right = float(tai_right + prof_right)
                        nb_pixel = input_features_np[i + 1, h, :]
                        if isEvent(nb_pixel) and abs(t_right - t_current) <= 500.0:
                            neighbors.append((base_name, t_right, float(lat_right), float(lon_right), h))

                    # Emit a single record for each event pixel:
                    # Key:   (base_name, t, lat, lon, h)
                    # Value: list of neighbor coords
                    results.append((key, neighbors))

    return results

def main(hdf5_chunks_dir, output_path):
    """
    PySpark job that:
      1) Lists .h5 chunk files in `hdf5_chunks_dir`.
      2) Parallelizes those chunk paths.
      3) For each chunk, creates adjacency for event pixels keyed by (fileName, t, lat, lon, h).
      4) Writes adjacency as text lines: 
         "(((fileName, t, lat, lon, h), [ (fileName, t2, lat2, lon2, h2), ... ]))"
      5) Removes output_path if it exists.
    """
    spark = SparkSession.builder.appName("BuildEventAdjacency").getOrCreate()
    sc = spark.sparkContext

    # Remove output directory if it exists
    if os.path.exists(output_path):
        print(f"[INFO] Removing existing directory: {output_path}")
        shutil.rmtree(output_path)

    # List chunked .h5 files
    chunk_files = [
        os.path.join(hdf5_chunks_dir, fname)
        for fname in os.listdir(hdf5_chunks_dir)
        if fname.endswith(".h5")
    ]
    if not chunk_files:
        print(f"No .h5 files found in {hdf5_chunks_dir}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    # Parallelize file list
    files_rdd = sc.parallelize(chunk_files)

    # Build adjacency for each chunk => RDD of ((fileName, t, x, y, h), neighbors_list)
    adjacency_rdd = files_rdd.flatMap(build_event_adjacency)

    # Save as text
    adjacency_rdd.saveAsTextFile(output_path)

    spark.stop()

if __name__ == "__main__":
    """
    Usage:
      spark-submit build_xyth_adjacency.py <chunks_dir> <output_dir>

    Example:
      spark-submit build_xyth_adjacency.py \
          ./chunks4spark \
          ./event_adjacency_out
    """
    if len(sys.argv) < 3:
        print("Usage: spark-submit build_xyth_adjacency.py <chunks_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    chunks_dir = sys.argv[1]
    out_dir = sys.argv[2]
    main(chunks_dir, out_dir)
