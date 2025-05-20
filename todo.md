# Todo List for Reproducing Paper Results

The paper: Refining epileptogenic high-frequency oscillations using deep learning: a reverse engineering approach

- [ ] **Task 1: Detailed Review of Paper for Methods and Results**
    - [ ] Carefully read the 'Methods' section of fcab267_text.txt to understand all procedures.
    - [ ] Carefully read the 'Results' section of fcab267_text.txt to identify all tables and figures to be reproduced.
    - [ ] List specific parameters for synthetic data generation (sampling rate, channels, event types, durations).
    - [ ] Document preprocessing steps (filtering, referencing, epoching, artifact rejection logic if any beyond the model).
    - [ ] Document feature engineering/representation (time-frequency plots, EEG trace plots, amplitude-coding plots - including generation parameters like wavelet type, frequency range, image dimensions).
    - [ ] Document machine learning model architectures (ResNet-18 modifications, input specifics for each of the two models, loss functions, optimizers, training parameters).
    - [ ] List specific performance metrics to be reported (accuracy, F1-score, AUC).
    - [ ] List specific visualizations to be generated (examples of HFOs, input feature plots, model performance plots, feature importance/interpretability plots if described).

- [ ] **Task 2: Generate Synthetic iEEG Data**
    - [ ] Implement Python code to generate synthetic iEEG data based on paper's description and user's preference (e.g., 32 channels, 2000 Hz sampling rate).
    - [ ] Simulate different event types: background EEG, HFOs (ripples 80-250Hz, fast ripples 250-500Hz), spikes, and combinations (spk-HFOs).
    - [ ] Create labels for these events (artefact, non-spk-HFO, spk-HFO).
    - [ ] Ensure data format is compatible with MNE-Python.

- [ ] **Task 3: Implement Preprocessing and Feature Generation**
    - [ ] Write Python code for iEEG data preprocessing using MNE-Python (e.g., filtering, referencing).
    - [ ] Implement the automated HFO detection logic (STE detector) or a simplified version if full reproduction is too complex, to identify candidate HFOs (c-HFOs).
    - [ ] Implement code to segment data into 1-second windows around c-HFOs.
    - [ ] Implement Python code to generate the three types of input images for the CNN from the 1-second windows:
        - [ ] Time-frequency plots (scalograms using Gabor wavelets, 10-500 Hz).
        - [ ] EEG tracing plots.
        - [ ] Amplitude-coding plots.
    - [ ] Ensure images are resized to 224x224 pixels.

- [ ] **Task 4: Implement Deep Learning Models (PyTorch)**
    - [ ] Implement the first CNN model (artefact detector) based on ResNet-18 in PyTorch.
        - [ ] Input: 3 identical time-frequency plots.
        - [ ] Output: Artefact vs. Real HFO.
    - [ ] Implement the second CNN model (spk-HFO detector) based on ResNet-18 in PyTorch.
        - [ ] Input: Concatenated time-frequency, EEG trace, and amplitude-coding plots.
        - [ ] Output: spk-HFO vs. non-spk-HFO.
    - [ ] Implement training loop, loss function (binary cross-entropy), and optimizer (Adam).
    - [ ] Prepare data loaders for training and testing.

- [ ] **Task 5: Train and Evaluate Models**
    - [ ] Train the artefact detector model on the synthetic dataset.
    - [ ] Evaluate its performance (accuracy, F1-score).
    - [ ] Train the spk-HFO detector model on the 'Real HFOs' identified by the first model (or synthetic labels).
    - [ ] Evaluate its performance (accuracy, F1-score).
    - [ ] If feasible, implement a simplified version of the eHFO/non-eHFO classification or the interpretability analysis (e.g., generating an 'inverted T-shape' plot).
 