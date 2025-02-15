import wfdb
import matplotlib.pyplot as plt
import numpy as np

def download_and_plot_ecg():
    """
    Downloads and plots a sample ECG recording from PhysioNet's MIT-BIH database
    """
    try:
        print("Downloading record 100 from MIT-BIH Arrhythmia Database...")
        # Download the record
        record = wfdb.rdrecord('100', pn_dir='mitdb')
        
        if record is None:
            raise Exception("Failed to load record")
            
        # Get the signal data
        signals = record.p_signal
        
        # Create time array (in seconds)
        time = np.arange(len(signals)) / record.fs
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot first lead of ECG (first 1000 samples)
        plt.plot(time[:1000], signals[:1000, 0])
        
        # Add labels and title
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.title('Sample ECG from MIT-BIH Database (Record 100)')
        plt.grid(True)
        
        # Show the plot
        plt.show()
        
        # Print recording information
        print("\nRecording Information:")
        print(f"Duration: {len(signals)/record.fs:.2f} seconds")
        print(f"Sampling frequency: {record.fs} Hz")
        print(f"Number of signals: {record.n_sig}")
        print(f"Signal names: {record.sig_name}")
        
        return record  # Return record for further analysis if needed
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify wfdb package is installed (pip install wfdb)")
        print("3. Try a different record number")
        return None

if __name__ == "__main__":
    print("Starting PhysioNet data access demo...")
    record = download_and_plot_ecg()