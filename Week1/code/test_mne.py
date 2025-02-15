import mne
import matplotlib.pyplot as plt

def simple_mne_demo():
    """
    Simple demonstration of MNE-Python capabilities using sample data
    """
    print("Loading sample data...")
    # Download and load sample data
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                           'sample_audvis_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file)
    
    # Print some basic information about the data
    print("\nDataset Information:")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Recording length: {raw.times.max():.2f} seconds")
    
    # Plot first 10 seconds of data
    print("\nPlotting first 10 seconds of EEG channels...")
    # Pick only EEG channels for clarity
    raw_eeg = raw.pick_types(meg=False, eeg=True)
    
  
    
    # Plot the first 10 seconds
    raw_eeg.plot(duration=10, n_channels=5, 
                 scalings='auto', title='Sample EEG Data')
    
    # Create a Power Spectral Density plot
    print("\nCreating PSD plot...")
    raw_eeg.plot_psd(fmax=50, average=True)
    
    plt.show()

if __name__ == "__main__":
    try:
        print("Starting MNE-Python demonstration...")
        simple_mne_demo()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if "sample data" in str(e).lower():
            print("\nTip: The first run might take longer as it downloads sample data.")