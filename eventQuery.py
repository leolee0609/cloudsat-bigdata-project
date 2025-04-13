"""
event_query_api_xyth_rounded.py

Ensures:
  arr[0] -> Latitude
  arr[1] -> Longitude
  arr[2] -> TAI_start
  arr[3] -> Profile_time
  ...
Now truly extends the footprint range on both ends beyond BFS-labeled footprints
for the 3D plot, with a 'show_3d_plots' toggle in query(...).
"""

import os
import glob
import ast
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

############################################################
# Approx. 300m in latitude => 0.0027 deg
############################################################
CELL_SIZE = 0.0027

def _roundedLatLon(lat, lon, cell=CELL_SIZE):
    """
    Round lat/lon to ~300m squares. 
    BFS-labeled data uses these for indexing footprints.
    """
    round_lat = int(round(lat / cell))
    round_lon = int(round(lon / cell))

    return (round_lat, round_lon)

class EventQueryAPI:
    def __init__(self, chunk_dir, event2pixels_dir):
        self.chunk_dir = chunk_dir
        self.event2pixels_dir = event2pixels_dir

        print("[INFO] Building chunk index map ...")
        self.chunk_index_map = self._build_chunk_index()

        print("[INFO] Loading event->pixels inverted index ...")
        self.event_to_pixels = self._load_event_inverted_index()

        print("[INFO] Initialization complete.")

    def _build_chunk_index(self):
        """
        For each chunk_*.h5, we read 'footprint_attributes' with columns:
          0=Latitude, 1=Longitude, 2=TAI_start, 3=Profile_time, ...
        We build a dict:
          chunk_index_map[chunkName][(t_val, lat_r, lon_r)] = row_idx
        so we can locate row indices for BFS-labeled footprints.
        """
        chunk_index_map = {}
        h5_files = glob.glob(os.path.join(self.chunk_dir, "chunk_*.h5"))
        for h5_path in h5_files:
            base_name = os.path.basename(h5_path)
            chunk_name = os.path.splitext(base_name)[0]

            with h5py.File(h5_path, "r") as hf:
                fpa = hf["footprint_attributes"]  
                # columns => [0=lat, 1=lon, 2=TAI_start, 3=Profile_time, ...]
                lat_vals = fpa[:, 0]
                lon_vals = fpa[:, 1]
                tai_vals = fpa[:, 2]
                prof_vals = fpa[:, 3]

                index_map = {}
                for i in range(fpa.shape[0]):
                    lat = float(lat_vals[i])
                    lon = float(lon_vals[i])
                    t_val = float(tai_vals[i] + prof_vals[i])
                    (lat_r, lon_r) = _roundedLatLon(lat, lon)
                    index_map[(t_val, lat_r, lon_r)] = i

            chunk_index_map[chunk_name] = index_map
        return chunk_index_map

    def _load_event_inverted_index(self):
        """
        BFS-labeled lines => (event_id, [(chunk_file, TAI_time, lat, lon, h_val), ...])

        We round lat/lon => store (chunk_file, TAI_time, lat_r, lon_r, h_val).
        This is consistent with the BFS-labeled snippet you showed, e.g.:
           (1340, [
             ('chunk_3750000_3800000.h5', 748525760.0, 2.2379510, 28.7316379, 66),
             ...
           ])
        """
        event_to_pixels = {}
        part_files = glob.glob(os.path.join(self.event2pixels_dir, "part-*"))
        for part_file in part_files:
            with open(part_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    (eid, px_list) = ast.literal_eval(line)
                    if eid not in event_to_pixels:
                        event_to_pixels[eid] = []
                    # BFS-labeled => (chunk_file, TAI_time, lat, lon, h_val)
                    for (chunk_file, t_val, lat, lon, h_val) in px_list:
                        if chunk_file.endswith(".h5"):
                            chunk_file = chunk_file[:-3]
                        (lat_r, lon_r) = _roundedLatLon(lat, lon)
                        # store => (chunk_file, TAI_time, lat_r, lon_r, h_val)
                        event_to_pixels[eid].append(
                            (chunk_file, float(t_val), lat_r, lon_r, int(h_val))
                        )

        return event_to_pixels

    def list_events(self):
        return sorted(self.event_to_pixels.keys())

    def query(self, event_id, modes=("2d",), extra_footprints=5, show_3d_plots=True):
        """
        Query footprints in '2d' or '3d' mode.

        If '3d' in modes, we load the multi-channel data for BFS-labeled footprints.
        Then, if show_3d_plots=True, we call _plot_3d_events(...) to display them.

        :param event_id: single event, list of events, or 'ALL'
        :param modes: e.g. ("2d") or ("2d","3d")
        :param extra_footprints: how many footprints to pad on left & right in the 3D plot
        :param show_3d_plots: if True, we do automatic multi-channel plots
        :return:
           - if "2d" only => a DataFrame
           - if "3d" => (DataFrame, feats_dict)
        """
        if event_id == "ALL":
            event_ids = self.list_events()
        elif isinstance(event_id, (list, tuple)):
            event_ids = event_id
        else:
            event_ids = [event_id]

        all_records = []
        feats_dict = {}

        for eid in event_ids:
            if eid not in self.event_to_pixels:
                print(f"[WARN] Event {eid} not found in index; skipping.")
                continue

            # BFS-labeled => (chunk_file, TAI_time, lat_r, lon_r, h_val)
            px_list = self.event_to_pixels[eid]
            # gather footprints needed => (chunk_name, TAI_time, lat_r, lon_r)
            footprints_needed = set((c, t, lr, lrn) for (c, t, lr, lrn, _) in px_list)

            footprints_per_chunk = {}
            for (chunk_name, t_val, lat_r, lon_r) in footprints_needed:
                chunk_map = self.chunk_index_map.get(chunk_name)
                if chunk_map is None:
                    continue
                row_idx = chunk_map.get((t_val, lat_r, lon_r))
                if row_idx is not None:
                    footprints_per_chunk.setdefault(chunk_name, []).append((t_val, lat_r, lon_r, row_idx))

            # 1) Load footprint_attributes => columns 0=Lat,1=Lon,2=TAI_start,3=Profile_time...
            event_records = []
            for chunk_name, rowlist in footprints_per_chunk.items():
                h5_path = os.path.join(self.chunk_dir, chunk_name + ".h5")
                with h5py.File(h5_path, "r") as hf:
                    fpa = hf["footprint_attributes"]
                    for (t_val, lat_r, lon_r, r_idx) in rowlist:
                        arr = fpa[r_idx]  # shape=(12,) presumably
                        # columns => 0=Latitude,1=Longitude,2=TAI_start,3=Profile_time,4=DEM,5=wind,6=Sigma0,7=NearSurfRef,8=SurfType,9=SurfHeightBin,10=LowestSigLayerTop,11=MODIS_scene_char
                        rec = {
                            "event_id": eid,
                            "chunk_file": chunk_name,
                            "t_val": t_val,
                            "lat_r": lat_r,
                            "lon_r": lon_r,
                            # Guarantee lat=arr[0], lon=arr[1]
                            "Latitude": arr[0],
                            "Longitude": arr[1],
                            "TAI_start": arr[2],
                            "Profile_time": arr[3],
                            "DEM_elevation": arr[4],
                            "Surface_wind": arr[5],
                            "Sigma_Zero": arr[6],
                            "Near_surface_reflectivity": arr[7],
                            "Surface_type": arr[8],
                            "SurfaceHeightBin": arr[9],
                            "Lowest_sig_layer_top": arr[10],
                            "MODIS_scene_char": arr[11],
                        }
                        event_records.append(rec)
            all_records.extend(event_records)

            # 2) If "3d", load input_features for BFS-labeled footprints
            if "3d" in modes:
                for chunk_name, rowlist in footprints_per_chunk.items():
                    h5_path = os.path.join(self.chunk_dir, chunk_name + ".h5")
                    with h5py.File(h5_path, "r") as hf:
                        inp_feats = hf["input_features"]
                        for (t_val, lat_r, lon_r, r_idx) in rowlist:
                            feats_2d = inp_feats[r_idx]
                            feats_dict[(eid, chunk_name, t_val, lat_r, lon_r)] = feats_2d

        # Return a DataFrame for 2D
        df = pd.DataFrame(all_records)

        if "3d" not in modes:
            return df

        # else => (df, feats_dict)
        if show_3d_plots:
            self._plot_3d_events(event_ids, df, feats_dict, extra_footprints=extra_footprints)
        return df, feats_dict

    def _plot_3d_events(self, event_ids, df, feats_dict, extra_footprints=5):
        """
        If show_3d_plots=True, we do the extended chunk approach:
         read entire chunk from [min_row - extra, max_row + extra], 
         then for each channel, plot BFS-labeled footprints with red squares.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        for eid in event_ids:
            px_list = self.event_to_pixels.get(eid, [])
            if not px_list:
                continue

            # BFS-labeled => row_idx
            chunk_to_rowids = {}
            for (chunk_name, t_val, lat_r, lon_r, h_val) in px_list:
                row_idx = self.chunk_index_map.get(chunk_name, {}).get((t_val, lat_r, lon_r))
                if row_idx is not None:
                    chunk_to_rowids.setdefault(chunk_name, []).append((row_idx, h_val))

            for chunk_name, row_hvals in chunk_to_rowids.items():
                if not row_hvals:
                    continue
                row_hvals.sort(key=lambda x: x[0])  # sort by row_idx
                row_indices = [rh[0] for rh in row_hvals]
                min_row = min(row_indices)
                max_row = max(row_indices)

                h5_path = os.path.join(self.chunk_dir, chunk_name + ".h5")
                with h5py.File(h5_path, "r") as hf:
                    n_foot = hf["footprint_attributes"].shape[0]
                    start_row = max(0, min_row - extra_footprints)
                    end_row   = min(n_foot, max_row + extra_footprints + 1)

                    feats_data = hf["input_features"]
                    all_row_idxs = list(range(start_row, end_row))
                    all_feats = [feats_data[r] for r in all_row_idxs]
                    if not all_feats:
                        continue

                    height, n_channels = all_feats[0].shape
                    row_index_to_col = {}
                    for col_idx, rowi in enumerate(all_row_idxs):
                        row_index_to_col[rowi] = col_idx

                    # For each channel => build matrix => highlight BFS-labeled
                    for c in range(n_channels):
                        mat = np.zeros((height, len(all_row_idxs)), dtype=np.float32)
                        for col_idx, feats2d in enumerate(all_feats):
                            mat[:, col_idx] = feats2d[:, c]

                        plt.figure(figsize=(8,6))
                        plt.title(f"Event {eid}, chunk={chunk_name}, channel {c}")
                        plt.xlabel(f"Footprint row {start_row}..{end_row-1}")
                        plt.ylabel("Height Bin")

                        plt.imshow(mat, aspect="auto", origin="upper")
                        plt.colorbar(label="Amplitude")

                        # BFS-labeled squares
                        for (ri, hv) in row_hvals:
                            if start_row <= ri < end_row:
                                colx = row_index_to_col[ri]
                                rect = Rectangle((colx - 0.5, hv - 0.5), 1, 1,
                                                 edgecolor="red", facecolor="none", linewidth=1)
                                plt.gca().add_patch(rect)

                        plt.show()
