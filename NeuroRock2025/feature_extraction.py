# -*- coding: utf-8 -*-
"""
NeuroRock EEG Data Analysis Script (Corrected v15 - Feature Extraction Shape Fix)

Extracts features from 1-second windows for classification analysis.
Loads data, creates initial 60s epochs with detailed metadata,
then creates 1s mini-epochs. Calculates features for each 1s window:
- Absolute & Relative Band Powers (Delta, Theta, Alpha, Beta, Gamma) for C3/C4
- Band Ratios (Alpha/Beta, Theta/Beta) for C3/C4
- Alpha Asymmetry Index
Saves the resulting features and metadata to a CSV file.

Usage:
    python feature_extraction.py --subject <SUBJECT_ID> [options]

Example:
    python feature_extraction.py --subject s01

Options:
    --fif-dir       Directory containing the .fif file (default: database)
    --ratings-dir   Directory containing the ratings .json file (default: ratings)
    --output-dir    Directory to save output feature file (default: .)
    --win_sec       Window length in seconds for mini-epochs (default: 1.0)
"""

import mne
import json
import numpy as np
import os
import matplotlib.pyplot as plt # Keep for potential future plots
import argparse
import pandas as pd
import warnings

# Configure MNE logging to be less verbose
mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", message="Channel locations not available. Disabling spatial colors.")

# --- Configuration ---
music_categories = {
    1: {"name": "High Arousal Positive", "short": "HAP", "clips": [1, 2, 3, 4]},
    2: {"name": "Low Arousal Positive", "short": "LAP", "clips": [5, 6, 7, 8]},
    3: {"name": "High Arousal Negative", "short": "HAN", "clips": [9, 10, 11, 12]},
    4: {"name": "Low Arousal Negative", "short": "LAN", "clips": [13, 14, 15, 16]}
}
FREQ_BANDS = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)}
FEATURE_BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] # Bands to extract features for
EPS = 1e-20 # Small constant

# --- Helper Functions ---
def get_category_from_clip_id(clip_id):
    for cat_id, category_info in music_categories.items():
        if clip_id in category_info["clips"]:
            return category_info["short"]
    return None

def load_and_create_long_epochs(fif_path, json_path):
    """Loads data, creates initial 60s epochs with detailed metadata."""
    tmin, tmax = 0.0, 60.0
    print(f"\nLoading data and creating epochs from {tmin}s to {tmax}s...")
    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True)
        print(f"Successfully loaded: {fif_path}"); print(f"Available channels: {raw.ch_names}")
        if 'stim' not in raw.ch_names: print("Error: 'stim' channel not found."); return None
        if 'C3' not in raw.ch_names or 'C4' not in raw.ch_names: print("Warning: C3 or C4 channel not found.")
    except Exception as e: print(f"Error loading FIF file: {e}"); return None
    ratings_data = None
    try:
        with open(json_path, 'r') as f: ratings_data = json.load(f)
        print(f"Successfully loaded: {json_path}"); ratings_data.sort(key=lambda x: x['timestamp'])
    except Exception as e: print(f"Error loading JSON file: {e}"); return None
    try: events = mne.find_events(raw, stim_channel='stim', initial_event=True, shortest_event=1); print(f"Found {len(events)} raw events.")
    except Exception as e: print(f"Error finding events: {e}"); return None
    start_music_events = events[events[:, 2] == 5]; print(f"Found {len(start_music_events)} 'start music' (Marker 5) events.")
    if len(start_music_events) == 0: print("Error: No 'start music' events."); return None
    if len(start_music_events) != len(ratings_data):
        print(f"Warning: Mismatched event/rating counts."); num_trials = min(len(start_music_events), len(ratings_data))
        start_music_events, ratings_data = start_music_events[:num_trials], ratings_data[:num_trials]; print(f"Proceeding with {num_trials} trials.")
    event_id_map = {"HAP": 1, "LAP": 2, "HAN": 3, "LAN": 4}
    new_event_list, metadata_list_for_long = [], []
    for i, event in enumerate(start_music_events):
        sample = event[0]; clip_info = ratings_data[i]; clip_id = clip_info['clip_id']; category = get_category_from_clip_id(clip_id)
        if category and category in event_id_map:
            category_event_id = event_id_map[category]; new_event_list.append([sample, 0, category_event_id])
            metadata_list_for_long.append({'event_id': category_event_id, 'category': category, 'clip_id': clip_id,
                                           'title': clip_info.get('title', 'Unknown'), 'artist': clip_info.get('artist', 'Unknown'),
                                           'original_event_sample': sample})
        else: print(f"Warning: Could not map clip_id {clip_id} to category.")
    if not new_event_list: print("Error: No valid events created."); return None
    categorized_events = np.array(new_event_list, dtype=int); initial_metadata = pd.DataFrame(metadata_list_for_long)
    print(f"Created {len(categorized_events)} categorized events with initial metadata.")
    try:
        long_epochs = mne.Epochs(raw, events=categorized_events, event_id=event_id_map, tmin=tmin, tmax=tmax, baseline=None,
                                 picks=['C3', 'C4'], metadata=initial_metadata, preload=True, event_repeated='drop')
        if len(long_epochs) == 0: print("Warning: No 60s epochs remaining."); return None
        if 'metadata' not in long_epochs.info or long_epochs.metadata is None or len(long_epochs.metadata) != len(long_epochs):
             print("Warning: Metadata may be inconsistent after epoch dropping.")
             if hasattr(long_epochs, 'selection') and initial_metadata is not None:
                  try:
                       valid_indices = [idx for idx in long_epochs.selection if idx < len(initial_metadata)]
                       long_epochs.metadata = initial_metadata.iloc[valid_indices].reset_index(drop=True)
                       if len(long_epochs.metadata) != len(long_epochs): long_epochs.metadata = None
                  except Exception as meta_e: print(f"Error re-indexing metadata: {meta_e}"); long_epochs.metadata = None
             else: long_epochs.metadata = None
        print(f"Successfully created initial epochs object ({tmin}s to {tmax}s) with {len(long_epochs)} epochs.");
        if long_epochs.metadata is not None: print("Metadata head:\n", long_epochs.metadata.head())
        return long_epochs
    except Exception as e: print(f"Error creating 60s epochs object: {e}"); return None


def create_short_epochs_from_long(long_epochs, duration=1.0, overlap=0.0):
    """Creates short, fixed-length epochs from longer epochs, preserving metadata."""
    all_short_epochs_list = []
    if long_epochs is None or long_epochs.metadata is None: print("Error: Valid long_epochs object required."); return None
    unique_clips = long_epochs.metadata['clip_id'].unique(); print(f"\nCreating {duration}s mini-epochs for {len(unique_clips)} unique clips...")
    original_event_id_map = long_epochs.event_id
    for clip_id in unique_clips:
        clip_metadata_rows = long_epochs.metadata[long_epochs.metadata['clip_id'] == clip_id]
        if clip_metadata_rows.empty: continue
        epoch_indices = long_epochs.metadata.index[long_epochs.metadata['clip_id'] == clip_id].tolist()
        if not epoch_indices: continue
        epoch_idx = epoch_indices[0]
        try:
            song_epoch = long_epochs[epoch_idx]
            song_metadata = clip_metadata_rows.iloc[[0]]
            data = song_epoch.get_data(copy=False)[0]; info = song_epoch.info
            raw_temp = mne.io.RawArray(data, info, first_samp=0)
            short_epochs_for_song = mne.make_fixed_length_epochs(raw_temp, duration=duration, overlap=overlap, preload=True)
            if len(short_epochs_for_song) == 0: continue
            n_short = len(short_epochs_for_song); metadata_list = []
            original_event_id_val = song_metadata['event_id'].iloc[0]; category_label = song_metadata['category'].iloc[0]
            base_meta_dict = song_metadata.iloc[0].to_dict()
            for i in range(n_short):
                 win_start = i * (duration - overlap)
                 meta_row = base_meta_dict.copy()
                 meta_row['window_start_time'] = win_start; meta_row['window_number'] = i
                 metadata_list.append(meta_row)
            short_epochs_for_song.metadata = pd.DataFrame(metadata_list)
            short_epochs_for_song.events[:, 2] = original_event_id_val
            short_epochs_for_song.event_id = {category_label: original_event_id_val}
            all_short_epochs_list.append(short_epochs_for_song)
        except Exception as e: print(f"Error creating short epochs for clip {clip_id} (index {epoch_idx}): {e}")
    if not all_short_epochs_list: print("Error: Failed to create short epochs."); return None
    final_short_epochs = mne.concatenate_epochs(all_short_epochs_list); final_short_epochs.event_id = original_event_id_map
    print(f"\nConcatenated short epochs. Total: {len(final_short_epochs)}."); print(f"Metadata includes: {list(final_short_epochs.metadata.columns)}")
    return final_short_epochs


def extract_features_per_window(short_epochs):
    """Calculates features for each 1-second epoch."""
    print(f"\nExtracting features for {len(short_epochs)} epochs...")
    if short_epochs is None or len(short_epochs) == 0: return None

    features = {}
    sfreq = short_epochs.info['sfreq']
    n_fft = int(sfreq); n_per_seg = n_fft # Use 1s window for PSD calc

    # --- Calculate PSD ---
    try:
        psds = short_epochs.compute_psd(method='welch', fmin=FREQ_BANDS['Delta'][0], fmax=FREQ_BANDS['Gamma'][1],
                                        picks=['C3', 'C4'], n_fft=n_fft, n_per_seg=n_per_seg, n_jobs=1, average=False)
        # SQUEEZE the potential extra dimension here!
        psds_data = psds.get_data().squeeze()
        freqs = psds.freqs
        # Check shape AFTER squeeze - should be (n_epochs, n_channels, n_freqs)
        if psds_data.ndim != 3:
             print(f"Error: PSD data has unexpected shape after squeeze: {psds_data.shape}. Expected 3 dimensions.")
             return None
        print(f"PSD calculated with shape: {psds_data.shape}")
    except Exception as e:
        print(f"Error computing PSD: {e}")
        return None

    # --- Extract Features ---
    ch_names = short_epochs.ch_names
    try: c3_idx = ch_names.index('C3'); c4_idx = ch_names.index('C4')
    except ValueError: print("Error: C3 or C4 not found."); return None

    # Calculate total power (axis 2 is frequencies)
    total_power_c3 = psds_data[:, c3_idx, :].sum(axis=1)
    total_power_c4 = psds_data[:, c4_idx, :].sum(axis=1)

    band_powers = {}
    # 1. Absolute and Relative Band Power
    for band in FEATURE_BANDS:
        fmin, fmax = FREQ_BANDS[band]
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        if not freq_mask.any(): print(f"Warn: No freqs for {band}."); continue

        # Absolute power: Average PSD values within the band (axis 1 is freqs AFTER indexing)
        abs_power_c3 = psds_data[:, c3_idx, freq_mask].mean(axis=1)
        abs_power_c4 = psds_data[:, c4_idx, freq_mask].mean(axis=1)

        # Ensure results are 1D
        if abs_power_c3.ndim != 1 or abs_power_c4.ndim != 1:
             print(f"Warn: Abs power calculation for {band} resulted in non-1D array. Skipping band.")
             continue # Skip this band if calculation failed

        features[f'Abs_{band}_C3'] = abs_power_c3
        features[f'Abs_{band}_C4'] = abs_power_c4
        band_powers[band] = {'C3': abs_power_c3, 'C4': abs_power_c4}

        # Relative power
        features[f'Rel_{band}_C3'] = (abs_power_c3 / (total_power_c3 + EPS)) * 100
        features[f'Rel_{band}_C4'] = (abs_power_c4 / (total_power_c4 + EPS)) * 100

    # 2. Band Ratios
    if 'Alpha' in band_powers and 'Beta' in band_powers:
        features['Ratio_AlphaBeta_C3'] = band_powers['Alpha']['C3'] / (band_powers['Beta']['C3'] + EPS)
        features['Ratio_AlphaBeta_C4'] = band_powers['Alpha']['C4'] / (band_powers['Beta']['C4'] + EPS)
    if 'Theta' in band_powers and 'Beta' in band_powers:
        features['Ratio_ThetaBeta_C3'] = band_powers['Theta']['C3'] / (band_powers['Beta']['C3'] + EPS)
        features['Ratio_ThetaBeta_C4'] = band_powers['Theta']['C4'] / (band_powers['Beta']['C4'] + EPS)

    # 3. Alpha Asymmetry
    if 'Alpha' in band_powers:
        power_c3 = band_powers['Alpha']['C3']
        power_c4 = band_powers['Alpha']['C4']
        features['Asym_Alpha'] = np.log(power_c4 + EPS) - np.log(power_c3 + EPS) # ln(R) - ln(L)

    # --- Combine with Metadata ---
    try:
        feature_df = pd.DataFrame(features) # Create DataFrame from features first
        if short_epochs.metadata is not None:
            metadata_df = short_epochs.metadata.reset_index(drop=True)
            # Check lengths before concat
            if len(metadata_df) == len(feature_df):
                 final_df = pd.concat([metadata_df, feature_df], axis=1)
            else:
                 print(f"Error: Metadata length ({len(metadata_df)}) != Feature length ({len(feature_df)}). Returning features only.")
                 final_df = feature_df # Fallback to features only
        else:
            print("Warning: No metadata found in epochs. Returning features only.")
            final_df = feature_df
    except ValueError as ve:
         # Catch the "Per-column arrays must each be 1-dimensional" error specifically
         print(f"Error creating DataFrame: {ve}")
         print("This likely means one of the calculated feature arrays was not 1D.")
         # Print shapes of feature arrays for debugging
         for key, val in features.items():
              print(f"  Shape of feature '{key}': {np.shape(val)}")
         return None
    except Exception as e:
        print(f"Error during DataFrame combination: {e}")
        return None


    print(f"Feature extraction complete. DataFrame shape: {final_df.shape}")
    print(f"Feature columns: {list(final_df.columns)}")
    return final_df


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features per window from NeuroRock EEG data (Corrected v15).")
    # Args remain the same
    parser.add_argument('--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--fif-dir', type=str, default='database', help="FIF file directory (default: 'database')")
    parser.add_argument('--ratings-dir', type=str, default='ratings', help="Ratings JSON directory (default: 'ratings')")
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for feature file (default: current directory)')
    parser.add_argument('--win_sec', type=float, default=1.0, help='Mini-epoch window length (sec) (default: 1.0)')
    args = parser.parse_args()

    fif_filename = os.path.join(args.fif_dir, f"{args.subject}_neurorock_eeg.raw.fif")
    json_filename = os.path.join(args.ratings_dir, f"{args.subject}_ratings.json")
    if not os.path.exists(fif_filename): print(f"Error: Input FIF file not found: {fif_filename}"); exit(1)
    if not os.path.exists(json_filename): print(f"Error: Input JSON file not found: {json_filename}"); exit(1)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True); print(f"Output features will be saved to: {output_dir}")

    print(f"\n--- Starting Feature Extraction for Subject: {args.subject} ---")
    long_epochs = load_and_create_long_epochs(fif_filename, json_filename)
    if long_epochs:
        short_epochs = create_short_epochs_from_long(long_epochs, duration=args.win_sec, overlap=0.0)
        if short_epochs:
            features_df = extract_features_per_window(short_epochs)
            if features_df is not None:
                feature_filename = os.path.join(output_dir, f"{args.subject}_features_win{args.win_sec:.1f}s.csv") # Format win_sec in filename
                try:
                    features_df.to_csv(feature_filename, index=False)
                    print(f"\nSuccessfully saved features to: {feature_filename}")
                    print("\n--- Feature Extraction Complete ---")
                except Exception as e:
                    print(f"\nError saving features to CSV: {e}")
                    print("\n--- Feature Extraction Encountered Errors ---")
            else: print("\n--- Feature Extraction Failed ---")
        else: print("\n--- Analysis Failed: Could not create short epochs ---")
    else: print("\n--- Analysis Failed: Could not create initial long epochs ---")