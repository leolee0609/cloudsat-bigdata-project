import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import datetime
import glob
import random
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------
# Modified Dataset Class
# ----------------------------------------

class PreprocessedSnowfallDataset(Dataset):
    def __init__(self, hdf5_file_path):
        """
        Args:
            hdf5_file_path (str): Path to the HDF5 file containing preprocessed data.
        """
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.num_samples = self.hdf5_file['input_features'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_features = self.hdf5_file['input_features'][idx]
        target_labels = self.hdf5_file['target_labels'][idx]
        attention_mask = self.hdf5_file['attention_mask'][idx]
        footprint_attributes = self.hdf5_file['footprint_attributes'][idx]

        # Convert to tensors
        input_features = torch.tensor(input_features, dtype=torch.float32)
        target_labels = torch.tensor(target_labels, dtype=torch.float32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        footprint_attributes = torch.tensor(footprint_attributes, dtype=torch.float32)

        return input_features, target_labels, attention_mask, footprint_attributes

# ----------------------------------------
# Original Dataset Class (for Preprocessing)
# ----------------------------------------

class SnowfallDataset(Dataset):
    def __init__(
        self,
        root_dir,
        profile_names,
        attribute_names,
        granule_list=None,  # Added granule_list parameter
        indices=None,
        pred_bins_num=4,
        fake_blind_zone_top_height=4,
        truncation_thickness=10,
        pad_value=-99999999.0,
        label_pad_value=-666666.0,
        print_log=False,
        snowfall_ratio=None,
        transform=None,  # Add transform function (e.g., log transform)
    ):
        # Initialization code
        self.root_dir = root_dir
        self.profile_names = profile_names
        self.attribute_names = attribute_names
        self.pred_bins_num = pred_bins_num
        self.fake_blind_zone_top_height = fake_blind_zone_top_height
        self.truncation_thickness = truncation_thickness
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value
        self.print_log = print_log
        self.snowfall_ratio = snowfall_ratio
        self.transform = transform

        # Load the list of granules
        self.granule_files = self._find_granules()


        # Limit granules if granule_list is provided
        if granule_list is not None:
            self.granule_files = [g for g in self.granule_files if g in granule_list]

        # Build an index mapping to efficiently access individual footprints
        self.index_mapping = self._build_index_mapping()


        # Subset the indices if provided
        if indices is not None:
            self.index_mapping = [self.index_mapping[i] for i in indices]

        # Initialize attribute cache
        self.attribute_cache = {}

        # Determine number of input features and footprint attributes
        sample_input_features, _, _, sample_footprint_attributes = self._get_sample(0, load_attributes=True)
        self.num_input_features = sample_input_features.shape[1]
        self.num_footprint_attributes = sample_footprint_attributes.shape[0]

    def _find_granules(self):
        """Find all granule files in the 'snowfall_rate' directory."""
        profile_dir = os.path.join(self.root_dir, "snowfall_rate")
        granules = [f[:-4] for f in os.listdir(profile_dir) if f.endswith('.npy')]
        return granules

    def _build_index_mapping(self):
        """Build an index mapping to access individual footprints across granules."""
        index_mapping = []
        total_samples = 0

        # Load or build granule info
        granule_info_file = os.path.join(self.root_dir, 'granule_info.pkl')
        if os.path.exists(granule_info_file):
            if self.print_log:
                print("Loading granule information from cache.")
            with open(granule_info_file, 'rb') as f:
                granule_info = pickle.load(f)
        else:
            granule_info = {}
            for granule in self.granule_files:
                snowfall_rate_path = os.path.join(self.root_dir, "snowfall_rate", f"{granule}.npy")
                snowfall_rate = np.load(snowfall_rate_path, mmap_mode='r')
                num_footprints = snowfall_rate.shape[0]
                granule_info[granule] = num_footprints
            with open(granule_info_file, 'wb') as f:
                pickle.dump(granule_info, f)

        for granule in self.granule_files:
            num_footprints = granule_info.get(granule, None)
            if num_footprints == None:
                continue
            for i in range(num_footprints):
                index_mapping.append((granule, i))
            total_samples += num_footprints

            if self.print_log:
                print(f"Granule '{granule}' has {num_footprints} footprints.")

        if self.print_log:
            print(f"Total samples in dataset: {total_samples}")

        return index_mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        return self._get_sample(idx)

    def _get_sample(self, idx, load_attributes=True):
        granule, footprint_idx = self.index_mapping[idx]

        if self.print_log:
            print(f"\nLoading sample index {idx}: Granule '{granule}', Footprint {footprint_idx}")

        # Load profiles for the selected granule and footprint
        profiles = {}
        for profile_name in self.profile_names:
            profile_path = os.path.join(self.root_dir, profile_name, f"{granule}.npy")
            profile_data = np.load(profile_path, mmap_mode='r')[footprint_idx]
            profiles[profile_name] = profile_data

        # Load attributes if required
        if load_attributes:
            attributes = self._get_attributes(granule, footprint_idx)
        else:
            attributes = None

        # Process the sample to generate inputs and targets
        input_features, target_labels, attention_mask, footprint_attributes = self._process_sample(
            profiles, attributes
        )

        # Convert to tensors
        input_features = torch.tensor(input_features, dtype=torch.float32)
        target_labels = torch.tensor(target_labels, dtype=torch.float32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        footprint_attributes = torch.tensor(footprint_attributes, dtype=torch.float32)

        return input_features, target_labels, attention_mask, footprint_attributes

    def _get_attributes(self, granule, footprint_idx):
        """Retrieve attributes for a given granule and footprint index."""
        if granule not in self.attribute_cache:
            attribute_path = os.path.join(self.root_dir, "attributes2d", f"{granule}_dataset2d.csv")
            attributes_df = pd.read_csv(attribute_path)
            self.attribute_cache[granule] = attributes_df
        else:
            attributes_df = self.attribute_cache[granule]

        attributes = {name: attributes_df.iloc[footprint_idx][name] for name in self.attribute_names}
        return attributes

    def _process_sample(self, profiles, attributes):
        """
        Process a single sample to generate input features, target labels, attention mask, and footprint attributes.
        """
        # Extract necessary attributes
        if attributes is not None:
            surface_bin = int(attributes['SurfaceHeightBin']) - 1  # Adjust to 0-based index
            surface_type = int(attributes['Surface_type'])
        else:
            # If attributes are not loaded, use default values
            surface_bin = profiles[self.profile_names[0]].shape[0] - 1  # Assume surface at the bottom bin
            surface_type = 1  # Default surface type

        # Determine near-surface bin based on surface type
        if surface_type in [7, 8]:
            near_surface_bin = surface_bin - 5
        else:
            near_surface_bin = surface_bin - 3

        # Define the fake blind zone
        fake_blind_zone_top_bin = near_surface_bin - self.fake_blind_zone_top_height + 1

        # Determine the range for selecting the chunk of target bins
        chunk_selection_end = fake_blind_zone_top_bin
        chunk_selection_start = max(chunk_selection_end - self.truncation_thickness, self.pred_bins_num)

        # Randomly select the base bin for the target chunk
        if chunk_selection_end - self.pred_bins_num >= chunk_selection_start:
            valid_start_bin = np.random.randint(chunk_selection_start, chunk_selection_end - self.pred_bins_num + 1)
        else:
            valid_start_bin = chunk_selection_start  # Use the start if range is invalid
        valid_end_bin = valid_start_bin + self.pred_bins_num

        # Prepare input features
        num_bins = profiles[self.profile_names[0]].shape[0]
        input_features = []
        attention_mask = []

        for bin_idx in range(num_bins):
            bin_features = []

            # Mask the target bins and fake blind zone in the snowfall_rate
            if bin_idx >= valid_start_bin and bin_idx < valid_end_bin:
                sfr_value = self.label_pad_value  # Masked value
            elif bin_idx >= fake_blind_zone_top_bin:
                sfr_value = self.pad_value  # Pad value for blind zone
            else:
                sfr_value = profiles['snowfall_rate'][bin_idx]

            # Replace negative values with 0.0
            if sfr_value < 0.0:
                sfr_value = 0.0

            # Collect auxiliary variables
            aux_values = []
            for aux_name in self.profile_names:
                if aux_name != 'snowfall_rate':
                    aux_val = profiles[aux_name][bin_idx]
                    if self.transform:
                        aux_val = self.transform(aux_val)
                    aux_values.append(aux_val)

            # Include positional information (bin_idx)
            position = bin_idx

            # Combine features
            bin_features = [sfr_value] + aux_values + [position]
            input_features.append(bin_features)

            # Create attention mask (True for valid inputs)
            if bin_idx >= fake_blind_zone_top_bin:
                attention_mask.append(False)
            else:
                attention_mask.append(True)

        input_features = np.array(input_features)

        # Extract target labels
        target_labels = profiles['snowfall_rate'][valid_start_bin:valid_end_bin].copy()
        # Replace negative snowfall rates with 0.0
        target_labels[target_labels < 0.0] = 0.0

        # Collect footprint attributes
        footprint_attributes = self._process_footprint_attributes(attributes)

        return input_features, target_labels, attention_mask, footprint_attributes

    def _process_footprint_attributes(self, attributes):
        """
        Process footprint attributes according to the specified requirements.
        """
        # Collect attributes in the required order
        attr_list = [
            attributes.get('Latitude', -9999.0),
            attributes.get('Longitude', -9999.0),
            attributes.get('TAI_start', -9999.0),
            attributes.get('Profile_time', -9999.0),
            attributes.get('DEM_elevation', -9999.0),
            attributes.get('Surface_wind', -9999.0),
            attributes.get('Sigma_Zero', -9999.0),
            attributes.get('Near_surface_reflectivity', -9999.0),
            attributes.get('Surface_type', -9999.0),
            attributes.get('SurfaceHeightBin', -9999.0),
            attributes.get('Lowest_sig_layer_top', -9999.0),
            attributes.get('MODIS_scene_char', -9999.0),
        ]

        footprint_attributes = np.array(attr_list, dtype=np.float32)
        return footprint_attributes

# ----------------------------------------
# Data Preprocessing Function
# ----------------------------------------

def preprocess_dataset(
    root_dir,
    profile_names,
    attribute_names,
    indices,
    output_path,
    pred_bins_num=4,
    fake_blind_zone_top_height=4,
    truncation_thickness=10,
    pad_value=-99999999.0,
    label_pad_value=-666666.0,
    print_log=False,
    transform=None,
    num_workers=4,
    granule_list=None,  # Added granule_list parameter
):
    """
    Preprocess the dataset and save to an HDF5 file.
    """
    dataset = SnowfallDataset(
        root_dir=root_dir,
        profile_names=profile_names,
        attribute_names=attribute_names,
        granule_list=granule_list,  # Pass granule_list to limit granules
        indices=indices,
        pred_bins_num=pred_bins_num,
        fake_blind_zone_top_height=fake_blind_zone_top_height,
        truncation_thickness=truncation_thickness,
        pad_value=pad_value,
        label_pad_value=label_pad_value,
        print_log=print_log,
        transform=transform,
    )

    # Set max sequence lengths
    max_seq_len = 125  # Adjust based on your data
    max_target_len = pred_bins_num  # Since target labels have length pred_bins_num
    print(f"Using max_seq_len: {max_seq_len}, max_target_len: {max_target_len}")

    # Prepare HDF5 file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as h5f:
        num_samples = len(dataset)
        # Preallocate datasets
        input_features_ds = h5f.create_dataset('input_features', (num_samples, max_seq_len, dataset.num_input_features), dtype=np.float32)
        target_labels_ds = h5f.create_dataset('target_labels', (num_samples, max_target_len), dtype=np.float32)
        attention_mask_ds = h5f.create_dataset('attention_mask', (num_samples, max_seq_len), dtype=bool)
        footprint_attributes_ds = h5f.create_dataset('footprint_attributes', (num_samples, dataset.num_footprint_attributes), dtype=np.float32)

        # Group indices by granule
        granule_to_indices = {}
        for idx in range(len(dataset)):
            granule, footprint_idx = dataset.index_mapping[idx]
            if granule not in granule_to_indices:
                granule_to_indices[granule] = []
            granule_to_indices[granule].append((idx, footprint_idx))

        # For each granule, process all required footprints
        total_samples_processed = 0
        for granule in tqdm(granule_to_indices.keys(), desc="Processing granules"):
            indices_in_granule = granule_to_indices[granule]
            # Load profiles for all required footprints in this granule
            profiles = {}
            for profile_name in dataset.profile_names:
                profile_path = os.path.join(root_dir, profile_name, f"{granule}.npy")
                profile_data = np.load(profile_path, mmap_mode='r')
                profiles[profile_name] = profile_data

            # Load attributes
            attribute_path = os.path.join(root_dir, "attributes2d", f"{granule}_dataset2d.csv")
            attributes_df = pd.read_csv(attribute_path)

            for idx, footprint_idx in indices_in_granule:
                # Get profiles for this footprint
                footprint_profiles = {name: profiles[name][footprint_idx] for name in dataset.profile_names}

                # Get attributes for this footprint
                attributes = attributes_df.iloc[footprint_idx][dataset.attribute_names].to_dict()

                # Process sample
                input_features, target_labels, attention_mask, footprint_attributes = dataset._process_sample(
                    footprint_profiles, attributes
                )

                # Pad sequences
                seq_len = input_features.shape[0]
                pad_len = max_seq_len - seq_len
                if pad_len > 0:
                    input_features = np.pad(input_features, ((0, pad_len), (0, 0)), mode='constant')
                    attention_mask = np.pad(attention_mask, (0, pad_len), mode='constant', constant_values=False)
                else:
                    input_features = input_features[:max_seq_len]
                    attention_mask = attention_mask[:max_seq_len]

                target_len = target_labels.shape[0]
                target_pad_len = max_target_len - target_len
                if target_pad_len > 0:
                    target_labels = np.pad(target_labels, (0, target_pad_len), mode='constant')
                else:
                    target_labels = target_labels[:max_target_len]

                # Save to HDF5 datasets
                input_features_ds[idx] = input_features
                target_labels_ds[idx] = target_labels
                attention_mask_ds[idx] = attention_mask
                footprint_attributes_ds[idx] = footprint_attributes

                total_samples_processed += 1

    print(f"Preprocessing completed and saved to {output_path}")

# ----------------------------------------
# DataLoader Creation Function
# ----------------------------------------

def create_preprocessed_dataloader(
    hdf5_file_path,
    batch_size,
    shuffle=True,
    num_workers=0,
):
    """
    Create a DataLoader for the preprocessed dataset.
    """
    dataset = PreprocessedSnowfallDataset(hdf5_file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader

# ----------------------------------------
# Data Preparation Function
# ----------------------------------------

def prepare_data(
    root_dir,
    profile_names,
    attribute_names,
    batch_num=5000,
    batch_size=256,
    save=True,
    saved_dataset_dir="./saved_datasets",
    random_seed=42,
    test_size=0.2,
    max_granules=None,  # Added max_granules parameter
):
    """
    Prepare training and validation datasets.
    """
    if not os.path.exists(saved_dataset_dir):
        os.makedirs(saved_dataset_dir)

    # List saved datasets
    saved_datasets = glob.glob(os.path.join(saved_dataset_dir, "*.pkl"))

    if saved_datasets:
        print("Existing saved datasets:")
        for i, dataset_file in enumerate(saved_datasets):
            print(f"{i}: {os.path.basename(dataset_file)}")
        selected = input("Select a dataset by number, or press Enter to create a new one: ")
        if selected.strip() != '':
            selected_idx = int(selected)
            dataset_file = saved_datasets[selected_idx]
            with open(dataset_file, 'rb') as f:
                indices_dict = pickle.load(f)
            return indices_dict['train_indices'], indices_dict['val_indices'], indices_dict['granules']
    # Else, create new dataset
    # Get list of all granules
    all_granules = [f[:-4] for f in os.listdir(os.path.join(root_dir, "snowfall_rate")) if f.endswith('.npy')]

    # Limit granules if max_granules is specified
    if max_granules is not None:
        selected_granules = all_granules[:max_granules]
    else:
        selected_granules = all_granules


    full_dataset = SnowfallDataset(
        root_dir=root_dir,
        profile_names=profile_names,
        attribute_names=attribute_names,
        granule_list=selected_granules,
        print_log=False,
    )

    # Create indices
    total_samples = len(full_dataset)
    indices = list(range(total_samples))


    # Shuffle indices
    random.seed(random_seed)
    random.shuffle(indices)

    # Split indices
    split_idx = int(np.floor(test_size * total_samples))
    train_indices_full, val_indices_full = indices[split_idx:], indices[:split_idx]

    # Select subset
    num_samples = batch_num * batch_size
    if num_samples > len(train_indices_full):
        num_samples = len(train_indices_full)
        print(f"Warning: Requested number of samples exceeds available training data. Using {num_samples} samples.")

    
    train_indices = train_indices_full[:num_samples]

    # Similarly for validation indices
    num_val_samples = int(num_samples * test_size / (1 - test_size))
    if num_val_samples > len(val_indices_full):
        num_val_samples = len(val_indices_full)
        print(f"Warning: Requested number of validation samples exceeds available validation data. Using {num_val_samples} samples.")

    val_indices = val_indices_full[:num_val_samples]

    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_file = os.path.join(saved_dataset_dir, f"dataset_{timestamp}.pkl")
        indices_dict = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'granules': selected_granules,  # Save the granules used
        }
        with open(dataset_file, 'wb') as f:
            pickle.dump(indices_dict, f)
        print(f"Saved dataset indices to {dataset_file}")

    return train_indices, val_indices, selected_granules


# ----------------------------------------
# Function to Create Training, Validation, and Test Sets
# ----------------------------------------

def create_train_val_test(dataloader, num_points, lat_range=(60, 80), lon_range=(60, 80), for_training=True, diff_height = False, fbz_h = 0):
    X_data, y_data, lats, lons = [], [], [], []

    X_fbz, y_fbz = [], []

    footprints_data, footprints_fbz = [], []

    thickness = 1
    if diff_height:
        thickness = 10

    for input_features, target_labels, attention_mask, footprint_attributes in dataloader:
        batch_size, seq_len, num_channels = input_features.size()

        for i in range(batch_size):
            lat, lon = footprint_attributes[i, 0].item(), footprint_attributes[i, 1].item()

            # Apply spatial filter for training and validation data
            if (lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]) == for_training:
                attention_mask_i = attention_mask[i]
                # Get the original x_past before any channel swapping
                x_past_all = input_features[i, attention_mask_i, :].clone().cpu().numpy()

                get_point = False
                for j in range(thickness):
                    baseIdx = x_past_all.shape[0] - j - fbz_h
                    x_past = x_past_all[: baseIdx].copy()
                    # Ensure there are enough time steps
                    if x_past.shape[0] < 9:
                        continue  # Skip if not enough data
    
                    # Get the label from the last 4 numbers in the first channel of x_past
                    label = x_past[-4:, 0].copy()
    
                    # Select the last 9 time steps
                    x_past = x_past[-9:, :]
                    # Swap snowfall rate (channel 0) with the last channel
                    x_past[:, [0, -1]] = x_past[:, [-1, 0]]
    
                    # Transpose and flatten
                    x_past = np.transpose(x_past)
                    x_past = x_past.flatten()
    
                    # Remove the last 4 elements (labels)
                    x_past = x_past[:-4].copy()
    
                    # Append to lists
                    X_data.append(x_past)
                    y_data.append(label)
                    lats.append(lat)
                    lons.append(lon)
                    footprints_data.append(footprint_attributes[i].tolist())
                    get_point = True

                if get_point:
                    # get the fbz point
                    x_past = x_past_all.copy()
                    # Ensure there are enough time steps
                    if x_past.shape[0] < 9:
                        continue  # Skip if not enough data
    
                    # Get the label from the last 4 numbers in the first channel of x_past
                    label = x_past[-4:, 0].copy()
    
                    # Select the last 9 time steps
                    x_past = x_past[-9:, :]
                    # Swap snowfall rate (channel 0) with the last channel
                    x_past[:, [0, -1]] = x_past[:, [-1, 0]]
    
                    # Transpose and flatten
                    x_past = np.transpose(x_past)
                    x_past = x_past.flatten()
    
                    # Remove the last 4 elements (labels)
                    x_past = x_past[:-4].copy()
    
                    # Append to lists
                    X_fbz.append(x_past)
                    y_fbz.append(label)
                    footprints_fbz.append(footprint_attributes[i].tolist())


                if len(X_data) % 500 == 0:
                    print(f'{len(X_data)}/{num_points} points gathered')
                    
                if len(X_data) >= num_points:
                    break
        if len(X_data) >= num_points:
            break

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    lats = np.array(lats)
    lons = np.array(lons)
    X_fbz, y_fbz, footprints_data, footprints_fbz = np.array(X_fbz), np.array(y_fbz), np.array(footprints_data), np.array(footprints_fbz)

    return X_data, y_data, X_fbz, y_fbz, lats, lons, footprints_data, footprints_fbz