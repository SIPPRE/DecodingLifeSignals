import pandas as pd
import numpy as np
import mne

# Path to your CSV file
csv_file = '/home/koutras/PAPEL/HMMY/Eggrafa/ERASMUS Coordinator/EUNICE Proposal/Decoding Life Signals/test_data/EEG_sample_recording.txt'

# Read CSV while skipping metadata lines and stripping extra spaces
df = pd.read_csv(csv_file, comment='%', skipinitialspace=True)
df.columns = df.columns.str.strip()

# Define the EXG channel names (assumed to be "EXG Channel 0" ... "EXG Channel 7")
eeg_channels = [f'EXG Channel {i}' for i in range(8)]

# Ensure the columns are numeric
for ch in eeg_channels:
    df[ch] = pd.to_numeric(df[ch], errors='coerce')

# Convert raw values to Volts:
# - The scaling factor 0.0223 converts raw counts to microvolts.
# - To convert microvolts to volts, multiply by 1e-6.
scaling_factor = 0.0223e-6  
eeg_data = df[eeg_channels].to_numpy() * scaling_factor

# Set the sampling frequency (from your metadata, here 250 Hz)
sfreq = 250

# Create MNE info structure (using 'eeg' channel type)
info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types=['eeg'] * len(eeg_channels))

# Create the MNE Raw object (data shape must be [n_channels, n_times])
raw = mne.io.RawArray(eeg_data.T, info)

# Rename the channels of the EEG sensors
rename_dict = {
    "EXG Channel 0": "Fz",
    "EXG Channel 1": "Cz",
    "EXG Channel 2": "Pz",
    "EXG Channel 3": "Oz",
    "EXG Channel 4": "T7",
    "EXG Channel 5": "T8",
    "EXG Channel 6": "P7",
    "EXG Channel 7": "P8",
}

raw.rename_channels(rename_dict)


# Apply a 50 Hz notch filter to remove line noise
raw.notch_filter(freqs=50)

# Plot the raw data (MNE will display amplitudes in µV by default)
raw.plot(block=True)
