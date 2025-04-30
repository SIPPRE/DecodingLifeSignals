# -*- coding: utf-8 -*-
"""
NeuroRock EEG Data Analysis Script (Corrected v10 - Song-by-Song Analysis)

Performs song-by-song analysis with 1-second resolution.
Loads data, creates initial 60s epochs with detailed metadata (clip_id, title),
then creates 1s mini-epochs. Generates visualizations for:
- Average PSD per song.
- Separate C3 & C4 frequency band power evolution within each song.
- Alpha asymmetry evolution within each song.

Usage:
    python analyze_neurorock_data.py --subject <SUBJECT_ID> [options]

Example:
    python analyze_neurorock_data.py --subject s01 --band Alpha

Options:
    --fif-dir       Directory containing the .fif file (default: database)
    --ratings-dir   Directory containing the ratings .json file (default: ratings)
    --output-dir    Directory to save output plots (default: ./{subject_id}_analysis_plots)
    --band          Frequency band to analyze for power evolution plot
                    (e.g., Alpha, Beta, Theta, Delta, Gamma). Default: Alpha.
                    Note: Asymmetry is always calculated using Alpha power if available.
    --win_sec       Window length in seconds for mini-epochs (default: 1.0)
"""

import mne
import json
import numpy as np
import os
import matplotlib.pyplot as plt
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
EPS = 1e-20 # Small constant to avoid log(0)

# --- Helper Functions ---
def get_category_from_clip_id(clip_id):
    for cat_id, category_info in music_categories.items():
        if clip_id in category_info["clips"]:
            return category_info["short"]
    return None

def load_and_create_long_epochs(fif_path, json_path):
    """Loads data, creates initial 60s epochs with detailed metadata."""
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
    tmin, tmax = 0.0, 60.0; baseline = None
    try:
        long_epochs = mne.Epochs(raw, events=categorized_events, event_id=event_id_map, tmin=tmin, tmax=tmax, baseline=baseline,
                                 picks=['C3', 'C4'], metadata=initial_metadata, preload=True, event_repeated='drop')
        if len(long_epochs) == 0: print("Warning: No 60s epochs remaining."); return None
        if 'metadata' not in long_epochs.info or long_epochs.metadata is None or len(long_epochs.metadata) != len(long_epochs):
             print("Warning: Metadata/epoch count mismatch after drops.") # MNE < 1.6 might handle this differently
             # Attempt to fix metadata if possible (example, may need adjustment based on MNE version)
             if hasattr(long_epochs, 'selection'):
                  try:
                       long_epochs.metadata = initial_metadata.iloc[long_epochs.selection].reset_index(drop=True)
                       print("Attempted to re-index metadata after drops.")
                       if len(long_epochs.metadata) != len(long_epochs):
                            print("Metadata re-indexing failed to match epoch count.")
                            long_epochs.metadata = None # Nullify if still mismatched
                  except Exception as meta_e:
                       print(f"Error re-indexing metadata: {meta_e}")
                       long_epochs.metadata = None


        print(f"Successfully created initial {tmax}s epochs object with {len(long_epochs)} epochs.");
        if long_epochs.metadata is not None:
             print("Metadata head:\n", long_epochs.metadata.head())
        else:
             print("Metadata is missing or inconsistent after epoch creation.")
        return long_epochs
    except Exception as e: print(f"Error creating 60s epochs object: {e}"); return None

def create_short_epochs_from_long(long_epochs, duration=1.0, overlap=0.0):
    """Creates short, fixed-length epochs from longer epochs, preserving metadata."""
    all_short_epochs_list = []
    if long_epochs is None or long_epochs.metadata is None: print("Error: Valid long_epochs object with metadata required."); return None
    unique_clips = long_epochs.metadata['clip_id'].unique(); print(f"\nCreating {duration}s mini-epochs for {len(unique_clips)} unique clips...")
    original_event_id_map = long_epochs.event_id
    for clip_id in unique_clips:
        # Handle potential multi-index if metadata was reset without drop=True
        clip_metadata_rows = long_epochs.metadata[long_epochs.metadata['clip_id'] == clip_id]
        if clip_metadata_rows.empty: continue
        # Get the index of the first row corresponding to this clip_id
        epoch_idx = clip_metadata_rows.index[0]

        try:
            song_epoch = long_epochs[epoch_idx] # Access epoch by its index
            song_metadata = clip_metadata_rows.iloc[[0]] # Get the corresponding single row of metadata

            data = song_epoch.get_data(copy=False)[0]; info = song_epoch.info
            raw_temp = mne.io.RawArray(data, info, first_samp=0)

            short_epochs_for_song = mne.make_fixed_length_epochs(raw_temp, duration=duration, overlap=overlap, preload=True)
            if len(short_epochs_for_song) == 0: continue
            n_short = len(short_epochs_for_song); metadata_list = []
            # Use .iloc[0] to get values from the single-row DataFrame song_metadata
            original_event_id_val = song_metadata['event_id'].iloc[0]; category_label = song_metadata['category'].iloc[0]
            base_meta_dict = song_metadata.iloc[0].to_dict() # Base metadata for this song

            for i in range(n_short):
                 win_start = i * (duration - overlap)
                 meta_row = base_meta_dict.copy() # Start with original song metadata
                 meta_row['window_start_time'] = win_start; meta_row['window_number'] = i
                 # event_id and category are already correct in base_meta_dict
                 metadata_list.append(meta_row)

            short_epochs_for_song.metadata = pd.DataFrame(metadata_list)
            short_epochs_for_song.events[:, 2] = original_event_id_val
            short_epochs_for_song.event_id = {category_label: original_event_id_val}
            all_short_epochs_list.append(short_epochs_for_song)
        except Exception as e: print(f"Error processing clip_id {clip_id} (index {epoch_idx}): {e}")

    if not all_short_epochs_list: print("Error: Failed to create short epochs."); return None
    final_short_epochs = mne.concatenate_epochs(all_short_epochs_list); final_short_epochs.event_id = original_event_id_map
    print(f"\nConcatenated short epochs. Total: {len(final_short_epochs)}."); print(f"Metadata includes: {list(final_short_epochs.metadata.columns)}")
    return final_short_epochs


# --- Plotting Functions ---
def plot_psd_per_song(epochs, output_dir, subject_id):
    """Calculates and plots the average PSD per song and saves the figure."""
    print("\nCalculating and plotting Power Spectral Density (PSD) per song...")
    if epochs is None or len(epochs) == 0 or epochs.metadata is None: print("No epochs/metadata."); return
    unique_clips = epochs.metadata['clip_id'].unique(); n_clips = len(unique_clips)
    if n_clips == 0: print("No unique clip IDs found."); return
    n_cols = 4; n_rows = (n_clips + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten(); plot_idx = 0
    clip_titles = dict(zip(epochs.metadata['clip_id'], epochs.metadata['title']))

    for clip_id in unique_clips:
        ax = axes_flat[plot_idx]
        try:
            song_epochs = epochs[f"clip_id == {clip_id}"]; n_song_epochs = len(song_epochs)
            if n_song_epochs > 0:
                 category = song_epochs.metadata['category'].iloc[0]; song_title = clip_titles.get(clip_id, f"ID {clip_id}")
                 psd_welch = song_epochs.compute_psd(method='welch', fmin=1.0, fmax=45.0, picks=['C3', 'C4'], n_jobs=1, average=False)
                 psd_data = psd_welch.get_data(); freqs = psd_welch.freqs
                 psd_avg_epochs = psd_data.mean(axis=0)
                 psd_avg_chans = psd_avg_epochs.mean(axis=0).squeeze() # Squeeze here
                 if psd_avg_chans.ndim == 1 and len(psd_avg_chans) == len(freqs):
                      ax.plot(freqs, 10 * np.log10(psd_avg_chans + EPS), label=f"{category}", linewidth=2)
                      ax.set_title(f"{song_title}\n({category}, N={n_song_epochs})", fontsize=9); ax.grid(True, linestyle=':')
                      if plot_idx >= (n_rows - 1) * n_cols: ax.set_xlabel("Frequency (Hz)")
                      if plot_idx % n_cols == 0: ax.set_ylabel("Power (dB/Hz)")
                 else: print(f"  Skipping plot clip {clip_id}: PSD shape mismatch."); ax.axis('off')
            else: ax.axis('off')
        except Exception as e: print(f"Error PSD clip {clip_id}: {e}"); ax.text(0.5, 0.5, 'Error', ha='center', va='center'); ax.axis('off')
        finally: plot_idx += 1
    while plot_idx < len(axes_flat): axes_flat[plot_idx].axis('off'); plot_idx += 1
    fig.suptitle(f'Avg PSD per Song (Mean over {epochs.tmax-epochs.tmin:.1f}s windows) - Subject {subject_id}', fontsize=14, y=1.0); fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    filename = os.path.join(output_dir, f"{subject_id}_psd_per_song_plot.png")
    try: fig.savefig(filename); print(f"Saved per-song PSD plot to: {filename}"); plt.show()
    except Exception as e: print(f"Error saving per-song PSD plot: {e}")
    finally: plt.close(fig)


def calculate_and_plot_band_power_evolution(short_epochs, band_name, band_freqs, output_dir, subject_id):
    """Calculates ABSOLUTE band power for C3 and C4 separately, adds to metadata, and plots evolution."""
    print(f"\nCalculating and plotting {band_name} ({band_freqs[0]}-{band_freqs[1]} Hz) power evolution (C3 vs C4)...")
    if short_epochs is None or len(short_epochs) == 0 or short_epochs.metadata is None: print("No short epochs/metadata."); return False
    unique_clips = sorted(short_epochs.metadata['clip_id'].unique()); n_clips = len(unique_clips)
    if n_clips == 0: print("No unique clip IDs found."); return False
    if 'C3' not in short_epochs.ch_names or 'C4' not in short_epochs.ch_names: print("C3/C4 missing."); return False

    metadata_col_c3 = f'{band_name}_power_C3'; metadata_col_c4 = f'{band_name}_power_C4'
    calculated_power = False
    try:
        psds = short_epochs.compute_psd(method='welch', fmin=band_freqs[0], fmax=band_freqs[1], picks=['C3', 'C4'], n_jobs=1, average=False)
        psds_data = psds.get_data()
        # Average across frequencies in the band (axis 2) and SQUEEZE
        band_power_per_epoch_ch = psds_data.mean(axis=2).squeeze() # Squeeze added here

        ch_names = short_epochs.ch_names; c3_idx = ch_names.index('C3'); c4_idx = ch_names.index('C4')
        # Check shape AFTER squeeze - should be (n_epochs, n_channels) = (960, 2)
        if band_power_per_epoch_ch.ndim == 2 and band_power_per_epoch_ch.shape[0] == len(short_epochs) and band_power_per_epoch_ch.shape[1] == 2:
             short_epochs.metadata[metadata_col_c3] = band_power_per_epoch_ch[:, c3_idx]
             short_epochs.metadata[metadata_col_c4] = band_power_per_epoch_ch[:, c4_idx]
             print(f"Calculated absolute {band_name} power for C3 and C4 for {len(short_epochs)} epochs.")
             calculated_power = True
        else:
             print(f"Error: Calculated band power shape {band_power_per_epoch_ch.shape} mismatch. Expected ({len(short_epochs)}, 2). Plotting aborted.")
             return False
    except ValueError: print("Error finding C3/C4 index."); return False
    except Exception as e: print(f"Error calculating PSD for band power: {e}"); return False

    # --- Plotting ---
    n_cols = 4; n_rows = (n_clips + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten(); plot_idx = 0
    clip_titles = dict(zip(short_epochs.metadata['clip_id'], short_epochs.metadata['title']))
    if metadata_col_c3 not in short_epochs.metadata or metadata_col_c4 not in short_epochs.metadata: print("Error: C3/C4 power columns missing."); plt.close(fig); return False

    power_c3_db = 10 * np.log10(short_epochs.metadata[metadata_col_c3] + EPS)
    power_c4_db = 10 * np.log10(short_epochs.metadata[metadata_col_c4] + EPS)
    global_min = min(power_c3_db.min(), power_c4_db.min()); global_max = max(power_c3_db.max(), power_c4_db.max())
    y_margin = (global_max - global_min) * 0.1 if (global_max - global_min) > 1e-6 else 1
    global_ylim = (global_min - y_margin, global_max + y_margin)

    for clip_id in unique_clips:
        ax = axes_flat[plot_idx]
        try:
            clip_metadata = short_epochs.metadata[short_epochs.metadata['clip_id'] == clip_id]
            if not clip_metadata.empty:
                 category = clip_metadata['category'].iloc[0]; song_title = clip_titles.get(clip_id, f"ID {clip_id}")
                 clip_metadata = clip_metadata.sort_values(by='window_start_time')
                 ax.plot(clip_metadata['window_start_time'], 10 * np.log10(clip_metadata[metadata_col_c3] + EPS), marker='.', linestyle='-', linewidth=1, label='C3', color='blue')
                 ax.plot(clip_metadata['window_start_time'], 10 * np.log10(clip_metadata[metadata_col_c4] + EPS), marker='.', linestyle='-', linewidth=1, label='C4', color='red')
                 ax.set_title(f"{song_title}\n({category})", fontsize=9); ax.grid(True, linestyle=':'); ax.set_ylim(global_ylim); ax.legend(loc='upper right', fontsize='x-small')
                 if plot_idx >= (n_rows - 1) * n_cols: ax.set_xlabel(f"Time (s)")
                 if plot_idx % n_cols == 0: ax.set_ylabel(f"{band_name} (dB)")
            else: ax.axis('off')
        except Exception as e: print(f"Error plotting band power clip {clip_id}: {e}"); ax.text(0.5, 0.5, 'Error', ha='center', va='center'); ax.axis('off')
        finally: plot_idx += 1
    while plot_idx < len(axes_flat): axes_flat[plot_idx].axis('off'); plot_idx += 1
    epoch_duration = short_epochs.tmax - short_epochs.tmin
    fig.suptitle(f'{band_name} Power Evolution (C3 vs C4) per Song ({epoch_duration:.1f}s windows) - Subject {subject_id}', fontsize=14, y=1.0); fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    filename = os.path.join(output_dir, f"{subject_id}_{band_name}_C3C4_evolution_plot.png")
    try: fig.savefig(filename); print(f"Saved {band_name} C3/C4 evolution plot to: {filename}"); plt.show()
    except Exception as e: print(f"Error saving {band_name} C3/C4 evolution plot: {e}")
    finally: plt.close(fig)
    return calculated_power


def plot_alpha_asymmetry(short_epochs, output_dir, subject_id):
    """Calculates and plots Alpha Asymmetry evolution within each song."""
    print(f"\nCalculating and plotting Alpha Asymmetry evolution...")
    power_col_c3 = 'Alpha_power_C3'; power_col_c4 = 'Alpha_power_C4'; asymmetry_col = 'alpha_asymmetry'
    if short_epochs is None or short_epochs.metadata is None or power_col_c3 not in short_epochs.metadata or power_col_c4 not in short_epochs.metadata:
        print(f"Error: Required Alpha power columns not found. Skipping asymmetry plot."); return
    unique_clips = sorted(short_epochs.metadata['clip_id'].unique()); n_clips = len(unique_clips)
    if n_clips == 0: print("No unique clip IDs found."); return
    try:
        power_c3 = short_epochs.metadata[power_col_c3].values; power_c4 = short_epochs.metadata[power_col_c4].values
        short_epochs.metadata[asymmetry_col] = np.log(power_c4 + EPS) - np.log(power_c3 + EPS) # ln(R) - ln(L)
        print(f"Calculated alpha asymmetry for {len(short_epochs)} epochs.")
    except Exception as e: print(f"Error calculating alpha asymmetry: {e}"); return
    n_cols = 4; n_rows = (n_clips + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten(); plot_idx = 0
    clip_titles = dict(zip(short_epochs.metadata['clip_id'], short_epochs.metadata['title']))
    global_min = short_epochs.metadata[asymmetry_col].min(); global_max = short_epochs.metadata[asymmetry_col].max()
    y_margin = (global_max - global_min) * 0.1 if (global_max - global_min) > 1e-6 else 0.5
    global_ylim = (global_min - y_margin, global_max + y_margin)

    for clip_id in unique_clips:
        ax = axes_flat[plot_idx]
        try:
            clip_metadata = short_epochs.metadata[short_epochs.metadata['clip_id'] == clip_id]
            if not clip_metadata.empty:
                 category = clip_metadata['category'].iloc[0]; song_title = clip_titles.get(clip_id, f"ID {clip_id}")
                 clip_metadata = clip_metadata.sort_values(by='window_start_time')
                 ax.plot(clip_metadata['window_start_time'], clip_metadata[asymmetry_col], marker='.', linestyle='-', linewidth=1, color='green')
                 ax.set_title(f"{song_title}\n({category})", fontsize=9); ax.grid(True, linestyle=':'); ax.set_ylim(global_ylim); ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
                 if plot_idx >= (n_rows - 1) * n_cols: ax.set_xlabel(f"Time (s)")
                 if plot_idx % n_cols == 0: ax.set_ylabel(f"Alpha Asymmetry")
            else: ax.axis('off')
        except Exception as e: print(f"Error plotting asymmetry clip {clip_id}: {e}"); ax.text(0.5, 0.5, 'Error', ha='center', va='center'); ax.axis('off')
        finally: plot_idx += 1
    while plot_idx < len(axes_flat): axes_flat[plot_idx].axis('off'); plot_idx += 1
    epoch_duration = short_epochs.tmax - short_epochs.tmin
    fig.suptitle(f'Alpha Asymmetry Evolution per Song ({epoch_duration:.1f}s windows) - Subject {subject_id}', fontsize=14, y=1.0); fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    filename = os.path.join(output_dir, f"{subject_id}_alpha_asymmetry_plot.png")
    try: fig.savefig(filename); print(f"Saved alpha asymmetry plot to: {filename}"); plt.show()
    except Exception as e: print(f"Error saving alpha asymmetry plot: {e}")
    finally: plt.close(fig)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NeuroRock EEG data song-by-song (Corrected v10).")
    # Args are the same
    parser.add_argument('--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--fif-dir', type=str, default='database', help="FIF file directory (default: 'database')")
    parser.add_argument('--ratings-dir', type=str, default='ratings', help="Ratings JSON directory (default: 'ratings')")
    parser.add_argument('--output-dir', type=str, default=None, help='Output plots directory (default: ./{subject_id}_analysis_plots)')
    parser.add_argument('--band', type=str, default='Alpha', choices=list(FREQ_BANDS.keys()), help=f"Freq band for C3/C4 power plot (default: Alpha)")
    parser.add_argument('--win_sec', type=float, default=1.0, help='Mini-epoch window length (sec) (default: 1.0)')
    args = parser.parse_args()

    fif_filename = os.path.join(args.fif_dir, f"{args.subject}_neurorock_eeg.raw.fif")
    json_filename = os.path.join(args.ratings_dir, f"{args.subject}_ratings.json")
    if not os.path.exists(fif_filename): print(f"Error: Input FIF file not found: {fif_filename}"); exit(1)
    if not os.path.exists(json_filename): print(f"Error: Input JSON file not found: {json_filename}"); exit(1)
    if args.output_dir is None: output_dir = f"{args.subject}_analysis_plots"
    else: output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True); print(f"Output plots will be saved to: {output_dir}")

    print(f"\n--- Starting Analysis for Subject: {args.subject} ---")
    long_epochs = load_and_create_long_epochs(fif_filename, json_filename)
    if long_epochs:
        short_epochs = create_short_epochs_from_long(long_epochs, duration=args.win_sec, overlap=0.0)
        if short_epochs:
            print("\n--- Generating Plots ---")
            # Plot PSD per Song
            plot_psd_per_song(short_epochs, output_dir, args.subject)

            # Calculate and Plot C3/C4 Power Evolution for selected band
            band_freqs = FREQ_BANDS[args.band]
            power_calculated = calculate_and_plot_band_power_evolution(
                short_epochs, args.band, band_freqs, output_dir, args.subject
            )

            # Plot Alpha Asymmetry
            # First, ensure Alpha power has been calculated and added to metadata
            alpha_power_calculated = False
            if f'Alpha_power_C3' in short_epochs.metadata and f'Alpha_power_C4' in short_epochs.metadata:
                 alpha_power_calculated = True
            elif args.band != 'Alpha': # If user selected another band, calculate Alpha now
                 print(f"\nCalculating Alpha power separately for asymmetry plot...")
                 alpha_freqs = FREQ_BANDS['Alpha']
                 # Call function just to calculate and add to metadata, maybe suppress plotting?
                 # For now, just rely on it adding metadata, plot will be generated too.
                 alpha_power_calculated = calculate_and_plot_band_power_evolution(
                     short_epochs, 'Alpha', alpha_freqs, output_dir, subject_id=f"{args.subject}_AlphaForAsym"
                 )

            if alpha_power_calculated:
                 plot_alpha_asymmetry(short_epochs, output_dir, args.subject)
            else:
                 print("\nCould not calculate/find Alpha power, skipping asymmetry plot.")


            print("\n--- Analysis Complete ---")
        else: print("\n--- Analysis Failed: Could not create short epochs ---")
    else: print("\n--- Analysis Failed: Could not create initial 60s epochs ---")