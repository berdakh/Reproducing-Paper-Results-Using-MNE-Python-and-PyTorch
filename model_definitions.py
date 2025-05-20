import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super(ModifiedResNet18, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if hasattr(models.ResNet18_Weights, 'DEFAULT') else True)

        # Modify the first convolutional layer if input_channels is not 3
        if input_channels != 3:
            original_conv1 = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(input_channels, original_conv1.out_channels, 
                                            kernel_size=original_conv1.kernel_size, 
                                            stride=original_conv1.stride, 
                                            padding=original_conv1.padding, 
                                            bias=original_conv1.bias)
            # Optional: Initialize the new conv1 weights, e.g., by averaging original weights if input_channels=1
            if input_channels == 1 and original_conv1.in_channels == 3:
                # Average the weights of the original 3 channels to initialize the new single channel
                original_weights = original_conv1.weight.data
                self.resnet.conv1.weight.data = original_weights.mean(dim=1, keepdim=True)

        # Get the number of input features for the classifier
        num_ftrs = self.resnet.fc.in_features

        # Replace the final fully connected layer with the custom layers as per the paper
        # Paper: "the last layer of the resnet18 was modified to be three fully connected layers
        # with LeakyReLU, BatchNorm and 10% dropout in between."
        # Assuming the output of these three layers then goes to a sigmoid for binary classification.
        # Let's define these layers. The dimensions are not specified, so we make a reasonable choice.
        # For example: num_ftrs -> 256 -> 64 -> num_classes
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes) # Output layer for binary classification (1 output unit)
        )
        # The sigmoid will be applied in the training loop or as part of the loss function (BCEWithLogitsLoss)

    def forward(self, x):
        return self.resnet(x)

def get_artefact_detector_model():
    """
    Returns the artefact detector model.
    Input: 3 identical time-frequency plots (concatenated as 3 channels or handled by input_channels=1 if grayscale).
    The paper states: "For the artefact detector, only the time-frequency information was used. 
    Hence time-frequency plots were repeated three times and concatenated together as the input to the artefact detector."
    So, input_channels = 3.
    Output: Probability of being an artefact (binary classification: Artefact vs Real HFO).
    """
    model = ModifiedResNet18(num_classes=1, input_channels=3)
    return model

def get_spk_hfo_detector_model():
    """
    Returns the spk-HFO detector model.
    Input: Concatenation of three feature-representing images (time-frequency, EEG tracing, amplitude-coding).
    These are likely grayscale images, stacked to form a 3-channel input.
    Output: Probability of being spk-HFO (binary classification: spk-HFO vs non-spk-HFO).
    """
    model = ModifiedResNet18(num_classes=1, input_channels=3)
    return model

if __name__ == '__main__':
    # Test the model instantiations
    print("Testing Artefact Detector Model instantiation...")
    artefact_model = get_artefact_detector_model()
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input_artefact = torch.randn(2, 3, 224, 224)
    try:
        output_artefact = artefact_model(dummy_input_artefact)
        print(f"Artefact Detector Output Shape: {output_artefact.shape}") # Expected: [2, 1]
    except Exception as e:
        print(f"Error during artefact model test: {e}")

    print("\nTesting Spk-HFO Detector Model instantiation...")
    spk_hfo_model = get_spk_hfo_detector_model()
    # Create a dummy input tensor
    dummy_input_spk_hfo = torch.randn(2, 3, 224, 224)
    try:
        output_spk_hfo = spk_hfo_model(dummy_input_spk_hfo)
        print(f"Spk-HFO Detector Output Shape: {output_spk_hfo.shape}") # Expected: [2, 1]
    except Exception as e:
        print(f"Error during spk-HFO model test: {e}")

    # Example of how the model might be used with BCEWithLogitsLoss
    # criterion = nn.BCEWithLogitsLoss()
    # target = torch.tensor([[0.], [1.]]) # Example target labels
    # loss = criterion(output_artefact, target)
    # print(f"\nExample loss calculation (artefact model): {loss.item()}")

