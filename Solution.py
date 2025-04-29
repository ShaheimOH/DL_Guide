import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

"""

Deep Learning for Atrial Fibrillation Classification Using Single-Lead ECG

This educational project introduces students to deep learning through a biomedical engineering lens. 

The objective is to build a binary classifier that distinguishes between normal heart rhythm and atrial fibrillation using one-lead ECG signals.

Important Note:
    
The dataset used in this task is based on real patient data but has been augmented for instructional purposes. 

IT IS NOT SUITABLE FOR ACADEMIC RESEARCH OR CLINICAL USE!

The data has also be normalised and preprocessed for direct deep learning use.

Labels:
  - 0 = Normal (Control)  
  - 1 = Atrial Fibrillation

Project Goals:
    - Design and implement a 1D Convolutional Neural Network (CNN) for binary classification.
    - Split the dataset into training and test sets (a validation set may also be used for hyperparameter tuning).
    - Evaluate model performance using test set accuracy and compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

The code provided below is a solution to this task.

"""
############# LOADING DATA #############

# loading data from numpy files

features = np.load('features.npy') # change the path to your features file
labels = np.load('labels.npy') # change the path to your labels file

# Convert to PyTorch tensors

features_tensor = torch.from_numpy(features).float()  # Convert features to float tensor
labels_tensor = torch.from_numpy(labels).long()       # Convert labels to long tensor (for classification tasks)

# Reshape features to (1516, 1, 18170) because 1D convolution expects 3D input (N, C, L). Link to documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

features_tensor = features_tensor.unsqueeze(1)  # Adds a dimension at position 1

# Check the dimensions of the tensors

print("Features tensor shape:", features_tensor.shape)
print("Labels tensor shape:", labels_tensor.shape)

# Creating pytorch dataloader

# splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

#Split into batches

batch_size = 10
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


############# DESIGNING CNN #############

# Let's define a simple Convolutional Neural Network (CNN) for binary classification
# We'll use nn.Module to create our own model class

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First 1D convolution layer
        # - Takes 1 input channel (because we have 1 feature channel)
        # - Outputs 16 channels (learns 16 different filters)
        # - Kernel size of 5 (filter length of 5)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        
        # Activation function (ReLU): introduces non-linearity
        self.relu1 = nn.ReLU()
        
        # MaxPooling layer
        # - Reduces dimensionality by taking max over a window of size 2
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second 1D convolution layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Flatten the output from conv layers to feed into fully connected (dense) layers
        # The size depends on input size, convolutions, and pooling. We'll calculate it dynamically later.

        # Fully connected (dense) layer
        self.fc1 = nn.Linear(32 * 4539, 64)  # 4539 is calculated based on input size and conv/pool ops
        
        # Output layer
        self.fc2 = nn.Linear(64, 2)  # Binary classification: output 2 classes (class 0 or 1)
        
    def forward(self, x):
        # Define how data flows through the model
        
        x = self.conv1(x)  # First conv
        x = self.relu1(x)  # Activation
        x = self.pool1(x)  # Max pooling
        
        x = self.conv2(x)  # Second conv
        x = self.relu2(x)  # Activation
        x = self.pool2(x)  # Max pooling
        
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        x = self.fc1(x)     # Fully connected layer
        x = self.fc2(x)     # Output layer
        return x

# Instantiate the model
model = SimpleCNN()

# Move model to device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

############# LOSS FUNCTION AND OPTIMIZER #############

# Loss function
# - We use CrossEntropyLoss for classification problems
criterion = nn.CrossEntropyLoss()

# Optimizer
# - Adam optimizer is a popular choice because it adapts learning rates automatically

optimizer = optim.Adam(model.parameters(), lr=0.001)

############# TRAINING THE MODEL #############

# Define number of epochs
num_epochs = 130

# Loop over epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    # Loop over batches
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass (compute gradients)
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.9f}")

############# TESTING THE MODEL #############

model.eval()  # Set model to evaluation mode (important: turns off dropout, etc.)

all_labels = []
all_probs = []  # We will store the probability for class 1

# We don't want gradients when evaluating
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)  # Raw logits (not probabilities)
        
        # Convert logits to probabilities using softmax
        probs = torch.softmax(outputs, dim=1)
        
        # We take probability of class 1 (positive class)
        class1_probs = probs[:, 1]
        
        all_labels.append(labels.cpu())
        all_probs.append(class1_probs.cpu())

# Concatenate all batches
all_labels = torch.cat(all_labels)
all_probs = torch.cat(all_probs)

# Calculate ROC AUC
roc_auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())

print(f"Test ROC AUC: {roc_auc:.4f}")