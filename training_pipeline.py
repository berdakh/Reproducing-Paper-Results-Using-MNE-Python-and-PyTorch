import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import os

# Assuming these files are in the same directory or project_files is in PYTHONPATH
# For direct execution, ensure paths are correct or add to sys.path
try:
    from synthetic_data_generator import generate_synthetic_ieeg_data
    from feature_extractor import extract_features_for_event, generate_time_frequency_plot, generate_eeg_tracing_plot, generate_amplitude_coding_plot
    from model_definitions import get_artefact_detector_model, get_spk_hfo_detector_model
except ImportError:
    print("Make sure synthetic_data_generator.py, feature_extractor.py, and model_definitions.py are accessible.")
    # As a fallback for notebook execution where files might be in a specific relative path:
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__))) # Add current script's directory
    from synthetic_data_generator import generate_synthetic_ieeg_data
    from feature_extractor import extract_features_for_event, generate_time_frequency_plot, generate_eeg_tracing_plot, generate_amplitude_coding_plot
    from model_definitions import get_artefact_detector_model, get_spk_hfo_detector_model

class HFOFeatureDataset(Dataset):
    """Dataset to load HFO features for PyTorch models."""
    def __init__(self, raw_data, sfreq, event_indices, model_type="artefact", window_duration=1.0):
        self.raw_data = raw_data
        self.sfreq = sfreq
        self.event_indices = event_indices # List of indices from raw_data.annotations
        self.model_type = model_type # "artefact" or "spk_hfo"
        self.window_duration = window_duration
        # Define the mapping from annotation description to numerical labels
        # For artefact detector: 0 for artefact, 1 for Real HFO (non-spk-HFO or spk-HFO)
        # For spk-HFO detector: 0 for non-spk-HFO, 1 for spk-HFO
        self.event_id_map_artefact = {"artefact".lower(): 0, "non-spk-HFO_ripple".lower(): 1, "non-spk-HFO_fr".lower(): 1, "spk-HFO".lower(): 1}
        self.event_id_map_spk_hfo = {"non-spk-HFO_ripple".lower(): 0, "non-spk-HFO_fr".lower(): 0, "spk-HFO".lower(): 1}

        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts numpy array (H x W x C) or (H x W) to (C x H x W) and scales to [0,1]
            # Add normalization if needed, based on training data stats
            # transforms.Normalize(mean=[0.5], std=[0.5]) # Example for single channel grayscale
        ])

    def __len__(self):
        return len(self.event_indices)

    def __getitem__(self, idx):
        event_original_idx = self.event_indices[idx]
        annotation_desc = self.raw_data.annotations.description[event_original_idx].lower()

        # Extract features (these are returned as HxW numpy arrays, grayscale)
        tf_img, eeg_img, amp_img, _ = extract_features_for_event(
            self.raw_data, event_original_idx, {}, # event_id_map not used for label here
            window_duration=self.window_duration, 
            sfreq=self.sfreq
        )

        if tf_img is None: # Handle cases where feature extraction failed
            # Return dummy data or raise an error / skip
            print(f"Warning: Feature extraction failed for event index {event_original_idx}. Returning zeros.")
            dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = torch.tensor([0.0], dtype=torch.float32)
            return dummy_img, dummy_label

        # Prepare model input based on model_type
        if self.model_type == "artefact":
            # Input: 3 identical time-frequency plots
            # tf_img is (224, 224), transform will make it (1, 224, 224)
            tf_tensor = self.transform(tf_img) # Becomes [1, 224, 224]
            model_input = torch.cat([tf_tensor, tf_tensor, tf_tensor], dim=0) # Becomes [3, 224, 224]
            label_val = self.event_id_map_artefact.get(annotation_desc, -1) # Default to -1 if not found
        elif self.model_type == "spk_hfo":
            # Input: Concatenated time-frequency, EEG trace, and amplitude-coding plots
            tf_tensor = self.transform(tf_img)
            eeg_tensor = self.transform(eeg_img)
            amp_tensor = self.transform(amp_img)
            model_input = torch.cat([tf_tensor, eeg_tensor, amp_tensor], dim=0) # Becomes [3, 224, 224]
            label_val = self.event_id_map_spk_hfo.get(annotation_desc, -1)
            # Filter out artefacts for spk-HFO model training
            if "artefact" in annotation_desc:
                # This sample should not be used for spk-HFO model, returning dummy that should be filtered
                # Or, filter event_indices in __init__ for spk_hfo model
                return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor([-1.0], dtype=torch.float32) # Special label to ignore
        else:
            raise ValueError("Invalid model_type")

        if label_val == -1:
            print(f"Warning: Annotation 	\"{annotation_desc}	\" not in map for model {self.model_type}. Event idx: {event_original_idx}")
            # This sample should be filtered out or handled
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor([-1.0], dtype=torch.float32)
            
        label = torch.tensor([float(label_val)], dtype=torch.float32)
        return model_input, label

def train_model(model, dataloader, criterion, optimizer, num_epochs=1, device="cpu"):
    """Simplified training loop."""
    model.to(device)
    model.train()
    print(f"Starting training for {num_epochs} epochs on {device}.")
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for i, (inputs, labels) in enumerate(dataloader):
            # Filter out samples with label -1 (e.g. artefacts for spk-HFO model, or unmapped labels)
            valid_indices = labels.squeeze() != -1.0
            if not torch.any(valid_indices):
                continue
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.nelement() == 0: # Skip if batch becomes empty
                continue

            # --- Debug: Print shape of inputs before model call ---
            # print(f"  Debug: inputs.shape before model call: {inputs.shape}, labels.shape: {labels.shape}")
            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # print(f"  Debug: Reshaping inputs from {inputs.shape} to 4D.")
                inputs = inputs.squeeze(1)
            # print(f"  Debug: inputs.shape after potential squeeze: {inputs.shape}")
            # --- End Debug ---

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels.bool()).sum().item()
            total_samples += labels.size(0)
            
            if (i + 1) % 10 == 0: # Print every 10 batches
                print(f"  Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    print("Training finished.")


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Parameters ---
    N_CHANNELS = 4 # Keep small for faster synthetic data generation for testing
    SFREQ = 2000
    DURATION_SEC = 30 # Shorter duration for quick test
    BATCH_SIZE = 8 # Smaller batch size for testing
    NUM_EPOCHS_TEST = 1 # Run only for 1 epoch for demonstration

    # 1. Generate Synthetic Data
    print("\n--- Generating Synthetic Data ---")
    raw_synthetic = generate_synthetic_ieeg_data(n_channels=N_CHANNELS, sfreq=SFREQ, duration_sec=DURATION_SEC)
    print(f"Generated {len(raw_synthetic.annotations)} annotations.")

    # Get all event indices
    all_event_indices = list(range(len(raw_synthetic.annotations)))
    if not all_event_indices:
        print("No events generated in synthetic data. Exiting.")
        exit()
        
    # Split indices for train/val (simple split for demo)
    # In practice, use patient-wise or proper stratified splitting
    np.random.shuffle(all_event_indices)
    split_idx = int(len(all_event_indices) * 0.8)
    train_event_indices = all_event_indices[:split_idx]
    val_event_indices = all_event_indices[split_idx:]

    # --- Artefact Detector Model Training (Example) ---
    print("\n--- Setting up Artefact Detector Model ---")
    artefact_train_dataset = HFOFeatureDataset(raw_synthetic, SFREQ, train_event_indices, model_type="artefact")
    # Filter out invalid samples from dataset (label -1)
    artefact_train_dataset.event_indices = [idx for idx in artefact_train_dataset.event_indices 
                                           if artefact_train_dataset.event_id_map_artefact.get(raw_synthetic.annotations.description[idx].lower(), -1) != -1]
    if not artefact_train_dataset.event_indices:
        print("No valid training samples for artefact detector after filtering. Skipping training.")
    else:
        artefact_train_dataloader = DataLoader(artefact_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        artefact_model = get_artefact_detector_model()
        criterion_artefact = nn.BCEWithLogitsLoss() # Handles sigmoid internally
        optimizer_artefact = optim.Adam(artefact_model.parameters(), lr=0.0003) # As per paper lr
        print("Starting Artefact Detector Training (DEMO - 1 epoch on small data)...")
        train_model(artefact_model, artefact_train_dataloader, criterion_artefact, optimizer_artefact, num_epochs=NUM_EPOCHS_TEST, device=DEVICE)

    # --- Spk-HFO Detector Model Training (Example) ---
    print("\n--- Setting up Spk-HFO Detector Model ---")
    # For spk-HFO, we only want non-artefact events
    real_hfo_event_indices_train = [idx for idx in train_event_indices 
                                    if "artefact" not in raw_synthetic.annotations.description[idx].lower() and 
                                       artefact_train_dataset.event_id_map_spk_hfo.get(raw_synthetic.annotations.description[idx].lower(), -1) != -1]
    
    if not real_hfo_event_indices_train:
        print("No valid training samples for Spk-HFO detector (non-artefacts with valid labels). Skipping training.")
    else:
        spk_hfo_train_dataset = HFOFeatureDataset(raw_synthetic, SFREQ, real_hfo_event_indices_train, model_type="spk_hfo")
        spk_hfo_train_dataloader = DataLoader(spk_hfo_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        spk_hfo_model = get_spk_hfo_detector_model()
        criterion_spk_hfo = nn.BCEWithLogitsLoss()
        optimizer_spk_hfo = optim.Adam(spk_hfo_model.parameters(), lr=0.0003)
        print("Starting Spk-HFO Detector Training (DEMO - 1 epoch on small data)...")
        train_model(spk_hfo_model, spk_hfo_train_dataloader, criterion_spk_hfo, optimizer_spk_hfo, num_epochs=NUM_EPOCHS_TEST, device=DEVICE)

    print("\n--- Main script finished ---")
    print("NOTE: This script provides a basic structure. Full-scale training requires more data, epochs, and proper validation.")

