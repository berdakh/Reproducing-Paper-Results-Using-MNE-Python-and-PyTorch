import numpy as np
import mne
from scipy import signal

def create_background_noise(n_channels, duration, sfreq):
    """Generates background pink noise."""
    # Generate Gaussian white noise
    noise = np.random.normal(0, 1, (n_channels, int(duration * sfreq)))
    # Simple approximation of pink noise by filtering white noise
    # Create a 1/f filter (more sophisticated methods exist but this is a start)
    # For simplicity, we'll use a lowpass filter to make it look more like EEG
    b, a = signal.butter(4, 30 / (sfreq / 2), btype='low')
    pink_noise = signal.lfilter(b, a, noise, axis=1)
    pink_noise *= 20 # Scale to a reasonable EEG amplitude (uV)
    return pink_noise

def create_hfo(sfreq, duration_ms, freq_band=(80, 250)):
    """Generates an HFO event (ripple or fast ripple)."""
    t = np.arange(0, duration_ms / 1000, 1/sfreq)
    # Create a sine wave in the middle of the band
    center_freq = np.mean(freq_band)
    hfo_signal = np.sin(2 * np.pi * center_freq * t)
    # Apply a Tukey window to make it a burst
    window = signal.windows.tukey(len(t), alpha=0.5)
    hfo_signal *= window
    # Filter to the specific band
    nyquist = sfreq / 2
    low = freq_band[0] / nyquist
    high = freq_band[1] / nyquist
    # Ensure high is less than 1.0 for the filter design
    if high >= 1.0:
        high = 0.99
    if low >= high:
        low = high - 0.01 # ensure low < high
        if low <=0:
            low = 0.01

    if low <= 0 or high <= 0 or low >=1 or high >=1 or low >= high:
        print(f"Skipping filter for HFO due to invalid band: {freq_band}")
    else:
        b, a = signal.butter(4, [low, high], btype='band')
        hfo_signal = signal.lfilter(b, a, hfo_signal)
    hfo_signal *= 10 # Amplitude in uV
    return hfo_signal

def create_spike(sfreq, duration_ms=50):
    """Generates a simple spike event."""
    t = np.arange(0, duration_ms / 1000, 1/sfreq)
    # A simple asymmetric triangular pulse
    spike_signal = np.zeros_like(t)
    peak_time = int(len(t) / 3)
    spike_signal[:peak_time] = np.linspace(0, 1, peak_time)
    spike_signal[peak_time:] = np.linspace(1, 0, len(t) - peak_time)
    spike_signal *= 50 # Amplitude in uV
    return spike_signal

def create_artefact(sfreq, duration_ms):
    """Generates a sharp transient artefact."""
    # Similar to a spike but could be sharper or different morphology
    t = np.arange(0, duration_ms / 1000, 1/sfreq)
    artefact_signal = np.zeros_like(t)
    peak_time = int(len(t) / 2)
    # A sharper, more symmetric pulse for an artefact
    if peak_time > 0 and len(t) - peak_time > 0:
        artefact_signal[:peak_time] = np.linspace(0, 1, peak_time)**2
        artefact_signal[peak_time:] = np.linspace(1, 0, len(t) - peak_time)**2
    artefact_signal *= 75 # Amplitude in uV
    return artefact_signal

def generate_synthetic_ieeg_data(n_channels=32, sfreq=2000, duration_sec=60):
    """Generates synthetic iEEG data with HFOs, spikes, and artefacts."""
    info = mne.create_info(
        ch_names=[f'EEG {i:03}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )
    data = create_background_noise(n_channels, duration_sec, sfreq)
    annotations_onset = []
    annotations_duration = []
    annotations_description = []

    # Add events
    n_events_per_type = 10 # Number of events of each type to add

    for i in range(n_events_per_type):
        # Add a non-spk-HFO (ripple)
        ch_idx = np.random.randint(0, n_channels)
        onset_sec = np.random.uniform(5, duration_sec - 5)
        hfo_dur_ms = np.random.uniform(60, 150)
        ripple = create_hfo(sfreq, hfo_dur_ms, freq_band=(80, 250))
        start_sample = int(onset_sec * sfreq)
        end_sample = start_sample + len(ripple)
        if end_sample < data.shape[1]:
            data[ch_idx, start_sample:end_sample] += ripple
            annotations_onset.append(onset_sec)
            annotations_duration.append(hfo_dur_ms / 1000)
            annotations_description.append('non-spk-HFO_ripple')

        # Add a non-spk-HFO (fast ripple)
        ch_idx = np.random.randint(0, n_channels)
        onset_sec = np.random.uniform(5, duration_sec - 5)
        hfo_dur_ms = np.random.uniform(60, 120)
        fast_ripple = create_hfo(sfreq, hfo_dur_ms, freq_band=(250, 500))
        start_sample = int(onset_sec * sfreq)
        end_sample = start_sample + len(fast_ripple)
        if end_sample < data.shape[1]:
            data[ch_idx, start_sample:end_sample] += fast_ripple
            annotations_onset.append(onset_sec)
            annotations_duration.append(hfo_dur_ms / 1000)
            annotations_description.append('non-spk-HFO_fr')

        # Add a spike
        ch_idx = np.random.randint(0, n_channels)
        onset_sec = np.random.uniform(5, duration_sec - 5)
        spike_dur_ms = np.random.uniform(30, 70)
        spike_event = create_spike(sfreq, spike_dur_ms)
        start_sample = int(onset_sec * sfreq)
        end_sample = start_sample + len(spike_event)
        if end_sample < data.shape[1]:
            data[ch_idx, start_sample:end_sample] += spike_event
            annotations_onset.append(onset_sec)
            annotations_duration.append(spike_dur_ms / 1000)
            annotations_description.append('spike') # Not directly a class in paper's DL, but good to have

        # Add a spk-HFO (spike + ripple)
        ch_idx = np.random.randint(0, n_channels)
        onset_sec = np.random.uniform(5, duration_sec - 5)
        hfo_dur_ms = np.random.uniform(60, 150)
        spike_dur_ms = np.random.uniform(30, 70)
        ripple_component = create_hfo(sfreq, hfo_dur_ms, freq_band=(80, 250))
        spike_component = create_spike(sfreq, spike_dur_ms)
        # Combine them, spike might slightly precede or overlap HFO
        combined_len = max(len(ripple_component), len(spike_component))
        spk_hfo_event = np.zeros(combined_len)
        # Place spike first, then HFO slightly overlapping or immediately after
        spike_start_idx = 0
        hfo_start_idx = int(len(spike_component) * 0.3) # HFO starts during the spike
        if spike_start_idx + len(spike_component) <= combined_len:
             spk_hfo_event[spike_start_idx : spike_start_idx+len(spike_component)] += spike_component
        if hfo_start_idx + len(ripple_component) <= combined_len:
            spk_hfo_event[hfo_start_idx : hfo_start_idx+len(ripple_component)] += ripple_component
        else: # if hfo is too long, truncate or place it differently
            hfo_start_idx = 0
            if hfo_start_idx + len(ripple_component) <= combined_len:
                 spk_hfo_event[hfo_start_idx : hfo_start_idx+len(ripple_component)] += ripple_component[:combined_len-hfo_start_idx]
            elif len(ripple_component) > 0:
                 spk_hfo_event[hfo_start_idx : combined_len] += ripple_component[:combined_len-hfo_start_idx]

        start_sample = int(onset_sec * sfreq)
        end_sample = start_sample + len(spk_hfo_event)
        if end_sample < data.shape[1] and len(spk_hfo_event) > 0:
            data[ch_idx, start_sample:end_sample] += spk_hfo_event
            annotations_onset.append(onset_sec)
            annotations_duration.append(len(spk_hfo_event) / sfreq)
            annotations_description.append('spk-HFO')

        # Add an artefact
        ch_idx = np.random.randint(0, n_channels)
        onset_sec = np.random.uniform(5, duration_sec - 5)
        artefact_dur_ms = np.random.uniform(20, 100)
        artefact_event = create_artefact(sfreq, artefact_dur_ms)
        start_sample = int(onset_sec * sfreq)
        end_sample = start_sample + len(artefact_event)
        if end_sample < data.shape[1]:
            data[ch_idx, start_sample:end_sample] += artefact_event
            annotations_onset.append(onset_sec)
            annotations_duration.append(artefact_dur_ms / 1000)
            annotations_description.append('artefact')

    annotations = mne.Annotations(annotations_onset, annotations_duration, annotations_description)
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(annotations)
    return raw

if __name__ == '__main__':
    # Parameters from paper/user
    N_CHANNELS = 32
    SFREQ = 2000  # Hz
    DURATION_SEC = 60 # Generate 1 minute of data for now

    print(f"Generating synthetic iEEG data: {N_CHANNELS} channels, {SFREQ} Hz, {DURATION_SEC} seconds...")
    raw_synthetic_data = generate_synthetic_ieeg_data(n_channels=N_CHANNELS, sfreq=SFREQ, duration_sec=DURATION_SEC)
    print("Synthetic data generated.")
    print(raw_synthetic_data)
    print(raw_synthetic_data.annotations)

    # Example: Save to a FIF file
    # output_path = "/home/ubuntu/project_files/synthetic_ieeg_raw.fif"
    # raw_synthetic_data.save(output_path, overwrite=True)
    # print(f"Synthetic data saved to {output_path}")

    # For the notebook, we'll likely call generate_synthetic_ieeg_data() directly.
    # This main block is for testing the script.
    pass
