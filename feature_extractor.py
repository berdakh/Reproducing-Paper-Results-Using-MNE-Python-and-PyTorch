import mne
import numpy as np
from scipy import signal
from skimage.transform import resize
import matplotlib.pyplot as plt
import io

# --- Helper function to convert matplotlib plot to numpy array ---
def plot_to_numpy_array(fig, target_size=(224, 224)):
    """Converts a matplotlib figure to a resized numpy array."""
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', bbox_inches='tight', pad_inches=0)
    io_buf.seek(0)
    img = plt.imread(io_buf)
    io_buf.close()
    plt.close(fig) # Close the figure to free memory
    # Convert to grayscale if it has an alpha channel (RGBA) or is RGB
    if img.ndim == 3 and img.shape[2] == 4: # RGBA
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) # Convert to grayscale
    elif img.ndim == 3 and img.shape[2] == 3: # RGB
        img = np.dot(img, [0.2989, 0.5870, 0.1140]) # Convert to grayscale
    # Resize
    img_resized = resize(img, target_size, anti_aliasing=True)
    return (img_resized * 255).astype(np.uint8) # Scale to 0-255

# --- Feature Generation Functions ---
def generate_time_frequency_plot(epoch_data, sfreq, target_size=(224, 224)):
    """Generates a time-frequency plot (scalogram) from epoch data."""
    freqs = np.logspace(*np.log10([10, 500]), num=100)
    n_cycles = freqs / 2.  # Different number of cycle per frequency
    
    if epoch_data.ndim == 1:
        epoch_data_mne = epoch_data.reshape(1, 1, -1)
    elif epoch_data.ndim == 2 and epoch_data.shape[0] == 1:
        epoch_data_mne = epoch_data.reshape(1, 1, -1)
    else:
        print("Warning: generate_time_frequency_plot received multi-channel data, using first channel.")
        epoch_data_mne = epoch_data[0,:].reshape(1, 1, -1)

    try:
        power = mne.time_frequency.tfr_morlet(mne.EpochsArray(epoch_data_mne, mne.create_info(['CH1'], sfreq, 'eeg'), tmin=-0.5, baseline=None, verbose=False),
                                                freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                return_itc=False, decim=1, n_jobs=1, verbose=False)
        fig, ax = plt.subplots(1, 1, figsize=(2.24, 2.24))
        power.plot(picks=[0], baseline=None, mode='logratio', axes=ax, colorbar=False, show=False, verbose=False)
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0)
        return plot_to_numpy_array(fig, target_size)
    except Exception as e:
        print(f"Error generating TFR plot: {e}")
        return np.zeros(target_size, dtype=np.uint8)

def generate_eeg_tracing_plot(epoch_data_segment, sfreq, target_size=(224, 224)):
    """Generates an EEG tracing plot from a 1-second epoch segment."""
    if epoch_data_segment.ndim > 1:
        trace_data = epoch_data_segment[0, :]
    else:
        trace_data = epoch_data_segment
        
    times = np.arange(len(trace_data)) / sfreq
    fig, ax = plt.subplots(1, 1, figsize=(2.24, 2.24))
    ax.plot(times, trace_data, color='black')
    ax.axis('off')
    fig.tight_layout(pad=0)
    return plot_to_numpy_array(fig, target_size)

def generate_amplitude_coding_plot(epoch_data_segment, target_size=(224, 224)):
    """Generates an amplitude-coding plot."""
    if epoch_data_segment.ndim > 1:
        signal_data = epoch_data_segment[0, :]
    else:
        signal_data = epoch_data_segment

    normalized_signal = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-9)
    img_strip = np.tile(normalized_signal, (target_size[0], 1))
    img_resized = resize(img_strip, target_size, anti_aliasing=True)
    return (img_resized * 255).astype(np.uint8)

# --- Main Preprocessing and Feature Extraction Function ---
def extract_features_for_event(raw, event_idx, event_id_map, window_duration=1.0, sfreq=2000):
    """Extracts the three image features for a given event in MNE Raw object."""
    annotations = raw.annotations
    onset = annotations.onset[event_idx]
    description = annotations.description[event_idx]
    label = -1
    if 'artefact' in description.lower():
        label = 0
    elif 'non-spk-HFO' in description.lower():
        label = 1
    elif 'spk-HFO' in description.lower():
        label = 2
    
    tmin = onset - (window_duration / 2)
    tmax = onset + (window_duration / 2) - 1/sfreq
    
    start_sample = int(raw.time_as_index(tmin, use_rounding=True))
    stop_sample = int(raw.time_as_index(tmax, use_rounding=True))

    start_sample = max(0, start_sample)
    stop_sample = min(raw.n_times, stop_sample)
    
    expected_samples = int(window_duration * sfreq)
    if (stop_sample - start_sample) < expected_samples * 0.9:
        print(f"Warning: Event {event_idx} at {onset}s results in a too short window. Skipping.")
        return None, None, None, -1
        
    epoch_data_segment, _ = raw[0, start_sample:stop_sample]
    epoch_data_segment = epoch_data_segment.ravel()

    if len(epoch_data_segment) < expected_samples:
        padding = expected_samples - len(epoch_data_segment)
        epoch_data_segment = np.pad(epoch_data_segment, (0, padding), 'constant', constant_values=(0,0))
    elif len(epoch_data_segment) > expected_samples:
        epoch_data_segment = epoch_data_segment[:expected_samples]

    tf_plot = generate_time_frequency_plot(epoch_data_segment, sfreq)
    eeg_trace_plot = generate_eeg_tracing_plot(epoch_data_segment, sfreq)
    amp_code_plot = generate_amplitude_coding_plot(epoch_data_segment)
    
    return tf_plot, eeg_trace_plot, amp_code_plot, label

# --- Example Usage (for testing this script) ---
if __name__ == '__main__':
    try:
        from synthetic_data_generator import generate_synthetic_ieeg_data
        print("Generating sample synthetic data for feature extraction testing...")
        raw_synthetic = generate_synthetic_ieeg_data(n_channels=5, sfreq=2000, duration_sec=20)
        print("Synthetic data generated.")
        print(raw_synthetic.annotations)

        if len(raw_synthetic.annotations.description) > 0:
            event_id_map = {'artefact': 0, 'non-spk-HFO_ripple': 1, 'non-spk-HFO_fr':1, 'spk-HFO': 2}
            
            print("\nExtracting features for the first few events...")
            for i in range(min(3, len(raw_synthetic.annotations.description))):
                print(f"\nProcessing event {i}: {raw_synthetic.annotations.description[i]} at {raw_synthetic.annotations.onset[i]}s")
                tf_img, eeg_img, amp_img, label = extract_features_for_event(raw_synthetic, i, event_id_map, sfreq=raw_synthetic.info['sfreq'])
                
                if tf_img is not None:
                    print(f"  Label: {label}")
                    print(f"  TF image shape: {tf_img.shape}, dtype: {tf_img.dtype}, min: {tf_img.min()}, max: {tf_img.max()}")
                    print(f"  EEG trace image shape: {eeg_img.shape}, dtype: {eeg_img.dtype}, min: {eeg_img.min()}, max: {eeg_img.max()}")
                    print(f"  Amplitude code image shape: {amp_img.shape}, dtype: {amp_img.dtype}, min: {amp_img.min()}, max: {amp_img.max()}")
                else:
                    print("  Skipped event due to issues.")
        else:
            print("No annotations found in synthetic data to extract features from.")

    except ImportError:
        print("Could not import synthetic_data_generator. Make sure it is in the Python path.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")

