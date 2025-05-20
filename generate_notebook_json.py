import json
import os

# --- Helper to read file content into a list of strings (lines) ---
def read_code_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return [f"# Error: File not found at {filepath}\n"]
    with open(filepath, 'r') as f:
        return [line for line in f.readlines()] # Keep newlines by default with readlines()

notebook_cells = []

# --- Cell 1: Introduction ---
notebook_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Reproduction of fcab267 Paper Results\n\n",
        "This notebook aims to reproduce key methodologies described in the paper 'Refining epileptogenic high-frequency oscillations using deep learning: a reverse engineering approach' (fcab267) by Zhang et al., 2021. This involves generating synthetic iEEG data and applying a deep learning pipeline for HFO classification.\n\n",
        "## Overview\n\n",
        "The process involves several key stages, implemented in the following sections:\n\n",
        "1.  **Synthetic iEEG Data Generation**: Functions to create artificial iEEG signals, including background activity, High-Frequency Oscillations (HFOs), spikes, and artefacts. This data will serve as the input for our models.\n",
        "2.  **Feature Extraction**: Code to transform the raw synthetic iEEG data segments into image-based representations suitable for Convolutional Neural Network (CNN) input. This includes time-frequency plots, EEG waveform traces, and amplitude-coded plots, as detailed in the paper.\n",
        "3.  **Deep Learning Model Definition**: Implementation of the ResNet-18 based CNN architectures for the two-step classification process: an artefact detector and an spk-HFO (spike-associated HFO) detector.\n",
        "4.  **Training and Evaluation Pipeline**: A demonstration of how to set up PyTorch Datasets and DataLoaders, and a simplified training loop to train the models on the generated synthetic data. This includes a basic evaluation of accuracy.\n\n",
        "**Important Notes:**\n",
        "-   This notebook provides the core code structure and a demonstration of the pipeline using synthetic data. Full reproduction of the paper's results would necessitate access to the original clinical dataset, adherence to the specific patient-wise cross-validation schemes, extensive model training, and rigorous hyperparameter optimization.\n",
        "-   The user requested that computationally intensive code be runnable on their local machine. This notebook is structured to facilitate that. You can execute cells sequentially. The training part is kept minimal (1 epoch) for demonstration purposes but can be extended.\n",
        "-   The code from the individual Python scripts (`synthetic_data_generator.py`, `feature_extractor.py`, `model_definitions.py`, `training_pipeline.py`) developed previously has been integrated into the cells of this notebook."
    ]
})

# --- Cell 2: Setup and Common Imports ---
imports_code = """
# Standard library imports
import os
import io

# Third-party library imports
import numpy as np
import scipy
from scipy import signal # Specifically for signal processing tasks
import matplotlib.pyplot as plt

# MNE for EEG data handling and analysis
import mne

# PyTorch for deep learning models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

# Scikit-image for image transformations if not covered by torchvision
from skimage.transform import resize

# Configure MNE to be less verbose unless errors occur
mne.set_log_level('ERROR')

print("All libraries imported successfully.")
"""
notebook_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [s + '\n' for s in imports_code.split('\n')]
})

# --- Cell 3: Synthetic Data Generation ---
synthetic_data_code_path = "/home/ubuntu/project_files/synthetic_data_generator.py"
notebook_cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "## 1. Synthetic iEEG Data Generation\n\n",
        "The following code block contains the functions to generate synthetic intracranial EEG (iEEG) data. This data will include background brain activity, High-Frequency Oscillations (HFOs) of different types (ripples: 80-250 Hz, fast ripples: 250-500 Hz), spike events, and artefacts. The main function, `generate_synthetic_ieeg_data`, assembles these components into an MNE `Raw` object, complete with annotations for each event type. This synthetic data will serve as the input for our feature extraction and deep learning models, mimicking the characteristics described in the reference paper (e.g., sampling rate of 2000 Hz, specified number of channels)."
    ]
})
raw_synthetic_code = read_code_from_file(synthetic_data_code_path)
# Remove the if __name__ == '__main__': block for notebook cell
cleaned_synthetic_code = []
in_main_block = False
for line in raw_synthetic_code:
    if line.strip() == "if __name__ == '__main__':":
        in_main_block = True
    if not in_main_block:
        cleaned_synthetic_code.append(line)
    # If the line is an import already covered, or a print statement from the original main, skip
    if "import mne" in line or "import numpy as np" in line or "from scipy import signal" in line:
        continue # Already in common imports
    if line.strip().startswith('print(f"Generating synthetic iEEG data') or line.strip().startswith('print("Synthetic data generated.")'):
        continue

notebook_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": cleaned_synthetic_code
})

# --- Cell 4: Example of Generating Synthetic Data ---
notebook_cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "### Example: Generating and Visualizing Synthetic Data\n\n",
        "Let's generate a short segment of synthetic data and inspect its properties and annotations."
    ]
})
synthetic_example_code = """
# Parameters for synthetic data generation (can be adjusted)
N_CHANNELS_SYNTH = 4  # Number of EEG channels
SFREQ_SYNTH = 2000    # Sampling frequency in Hz
DURATION_SEC_SYNTH = 10 # Duration of the recording in seconds

print(f"Generating synthetic iEEG data: {N_CHANNELS_SYNTH} channels, {SFREQ_SYNTH} Hz, {DURATION_SEC_SYNTH} seconds...")
raw_synthetic = generate_synthetic_ieeg_data(n_channels=N_CHANNELS_SYNTH, sfreq=SFREQ_SYNTH, duration_sec=DURATION_SEC_SYNTH)
print("Synthetic data generated successfully.")
print("\nRaw object info:")
print(raw_synthetic)
print("\nAnnotations present in the data:")
print(raw_synthetic.annotations)

# Optional: Plot a segment of the data
# raw_synthetic.plot(duration=5, n_channels=N_CHANNELS_SYNTH, scalings=dict(eeg=50e-6), title='Synthetic iEEG Data')
# plt.show() # You might need this in some environments to display the plot
"""
notebook_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [s + '\n' for s in synthetic_example_code.split('\n')]
})

# --- Cell 5: Feature Extraction ---
feature_extractor_code_path = "/home/ubuntu/project_files/feature_extractor.py"
notebook_cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "## 2. Feature Extraction\n\n",
        "The functions in this section are designed to convert segments of the iEEG data into image-based features. These features are the actual inputs to the Convolutional Neural Networks (CNNs). As per the paper, these include:\n",
        "-   **Time-Frequency Plots (Scalograms)**: Generated using continuous Gabor Wavelets (via MNE's Morlet TFR) to represent the signal's energy distribution across different frequencies over time.\n",
        "-   **EEG Tracing Plots**: Simple plots of the raw EEG waveform for a 1-second window around an event.\n",
        "-   **Amplitude-Coding Plots**: Images where pixel intensity in columns corresponds to the raw signal amplitude at each time point.\n\n",
        "The main function `extract_features_for_event` processes a detected event from the MNE `Raw` object and returns these three types of images, resized to 224x224 pixels."
    ]
})
raw_feature_code = read_code_from_file(feature_extractor_code_path)
cleaned_feature_code = []
in_main_block = False
for line in raw_feature_code:
    if line.strip() == "if __name__ == '__main__':":
        in_main_block = True
    if not in_main_block:
        # Skip imports already handled
        if line.strip().startswith("import mne") or line.strip().startswith("import numpy as np") or \ 
           line.strip().startswith("from scipy import signal") or line.strip().startswith("from skimage.transform import resize") or \ 
           line.strip().startswith("import matplotlib.pyplot as plt") or line.strip().startswith("import io") or \ 
           line.strip().startswith("from synthetic_data_generator import") :
            continue
        cleaned_feature_code.append(line)
notebook_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": cleaned_feature_code
})

# --- Cell 6: Example of Feature Extraction ---
notebook_cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "### Example: Extracting and Visualizing Features for an Event\n\n",
        "Here, we'll take the first annotated event from our synthetic data and generate its corresponding image features."
    ]
})
feature_example_code = """
if len(raw_synthetic.annotations.description) > 0:
    event_idx_example = 0 # Use the first event for demonstration
    print(f"Extracting features for event {event_idx_example}: {raw_synthetic.annotations.description[event_idx_example]} at {raw_synthetic.annotations.onset[event_idx_example]:.2f}s")
    
    # The event_id_map is not strictly needed by extract_features_for_event for label generation here, 
    # as the HFOFeatureDataset will handle mapping descriptions to numerical labels later.
    # We pass an empty dict for now.
    tf_img, eeg_img, amp_img, extracted_label_example = extract_features_for_event(
        raw_synthetic, 
        event_idx_example, 
        event_id_map={}, 
        sfreq=raw_synthetic.info['sfreq']
    )
    
    if tf_img is not None:
        print(f"  Extracted label based on description: {extracted_label_example}")
        print(f"  Time-Frequency Plot shape: {tf_img.shape}, dtype: {tf_img.dtype}")
        print(f"  EEG Trace Plot shape: {eeg_img.shape}, dtype: {eeg_img.dtype}")
        print(f"  Amplitude-Coding Plot shape: {amp_img.shape}, dtype: {amp_img.dtype}")
        
        # Visualize the extracted features
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Features for Event: {raw_synthetic.annotations.description[event_idx_example]}')
        axes[0].imshow(tf_img, cmap='gray', aspect='auto')
        axes[0].set_title('Time-Frequency Plot')
        axes[0].axis('off')
        axes[1].imshow(eeg_img, cmap='gray', aspect='auto')
        axes[1].set_title('EEG Trace Plot')
        axes[1].axis('off')
        axes[2].imshow(amp_img, cmap='gray', aspect='auto')
        axes[2].set_title('Amplitude-Coding Plot')
        axes[2].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        print(f"Could not extract features for event {event_idx_example}.")
else:
    print("No annotations in synthetic data to extract features from.")
"""
notebook_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [s + '\n' for s in feature_example_code.split('\n')]
})

# --- Cell 7: Model Definitions ---
model_definitions_code_path = "/home/ubuntu/project_files/model_definitions.py"
notebook_cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "## 3. Deep Learning Model Definitions\n\n",
        "This section provides the PyTorch implementation of the deep learning models. The architecture is based on ResNet-18, modified according to the paper's description for the two binary classification tasks:\n",
        "1.  **Artefact Detector**: Takes time-frequency plots as input (repeated three times to match 3-channel input of standard ResNet-18) and classifies events as either 'Artefact' or 'Real HFO'.\n",
        "2.  **Spk-HFO Detector**: Takes a 3-channel input composed of the time-frequency plot, EEG tracing plot, and amplitude-coding plot. It classifies 'Real HFOs' into 'spk-HFO' (HFO with spike) or 'non-spk-HFO' (HFO without spike)."
    ]
})
raw_model_code = read_code_from_file(model_definitions_code_path)
cleaned_model_code = []
in_main_block = False
for line in raw_model_code:
    if line.strip() == "if __name__ == '__main__':":
        in_main_block = True
    if not in_main_block:
        if line.strip().startswith("import torch") or line.strip().startswith("import torchvision.models as models"):
            continue
        cleaned_model_code.append(line)
notebook_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": cleaned_model_code
})

# --- Cell 8: Example of Model Instantiation ---
notebook_cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "### Example: Instantiating the Models"
    ]
})
model_example_code = """
# Instantiate the artefact detector
artefact_detector_model = get_artefact_detector_model()
print("Artefact Detector Model instantiated:")
# print(artefact_detector_model)

# Instantiate the spk-HFO detector
spk_hfo_detector_model = get_spk_hfo_detector_model()
print("\nSpk-HFO Detector Model instantiated:")
# print(spk_hfo_detector_model)

# Test with dummy input (optional, can be verbose)
# dummy_input = torch.randn(2, 3, 224, 224) # Batch of 2, 3 channels, 224x224
# try:
#     print(f"\nOutput shape from artefact model: {artefact_detector_model(dummy_input).shape}")
#     print(f"Output shape from spk-HFO model: {spk_hfo_detector_model(dummy_input).shape}")
# except Exception as e:
#     print(f"Error during model test with dummy input: {e}")
"""
notebook_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [s + '\n' for s in model_example_code.split('\n')]
})

# --- Cell 9: Training Pipeline (Dataset, DataLoader, Training Loop) ---
# The training_pipeline.py script contains the HFOFeatureDataset, train_model function, and the main execution block.
# We'll split these for clarity in the notebook.
training_pipeline_code_path = "/home/ubuntu/project_files/training_pipeline.py"
raw_training_code = read_code_from_file(training_pipeline_code_path)

# Extract HFOFeatureDataset class
hfo_dataset_class_code = []
class_started = False
class_braces = 0
for line in raw_training_code:
    if line.strip().startswith("class HFOFeatureDataset(Dataset):"):
        class_started = True
    if class_started:
        # Basic brace counting to find end of class (approximate for simple cases)
        # A proper AST parser would be better for robustness
        if "class " in line and not line.strip().startswith("class HFOFeatureDataset(Dataset):"):
             # another class started, so previous one ended
             if class_braces == 0: class_started = False # reset if it was a nested class definition somehow
        if class_started : hfo_dataset_class_code.append(line)
        if '{' in line: class_braces += line.count('{')
        if '}' in line: class_braces -= line.count('}')
        # Heuristic: class ends when back to global scope (indent 0) and not a comment/empty
        if not line.startswith(' ') and not line.startswith('\t') and line.strip() != "" and not line.strip().startswith("#") and "class HFOFeatureDataset" not in line:
            if class_braces == 0 : class_started = False 
            
# Extract train_
(Content truncated due to size limit. Use line ranges to read in chunks)