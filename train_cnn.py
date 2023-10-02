import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.optim as optim

class BallDetectionCNN(nn.Module):
    def __init__(self):
        super(BallDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max-pooling layers
        self.pool4 = nn.MaxPool2d(4, 4)  # Increase pooling size
        self.pool2 = nn.MaxPool2d(2, 2)  # Increase pooling size
        
        # Fully connected layers
        width = 600/2;
        height = 416/2;

        lin_size = int(width/2**3 * height/2**3 * 64)

        self.fc1 = nn.Linear(lin_size, 128)  # Adjust input size
        self.fc2 = nn.Linear(128, 8)  # 2 output classes (ball or not)

    def forward(self, x):
        x = self.pool2(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool2(torch.relu(self.conv3(x)))
        
        x = x.flatten()  # Flatten the feature maps
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, lbl_data_folder, img_data_folder, transform=None):
        self.lbl_data_folder = lbl_data_folder
        self.img_data_folder = img_data_folder
        self.transform = transform
        self.data_files = os.listdir(lbl_data_folder)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        lbl_filename = os.path.join(self.lbl_data_folder, self.data_files[idx])
        img_filename = os.path.join(self.img_data_folder, self.data_files[idx])

        lbl_data = torch.load(lbl_filename).reshape(1,8).squeeze()
        img_data = torch.load(img_filename)

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, lbl_data

class Uint8ToFloatTransform(object):
    def __call__(self, sample):
        return sample.float() / 255.0

class Reshape3DTransform(object):
    def __call__(self, sample):
        return sample.reshape(sample.shape[0], sample.shape[1], -1).permute(2, 0, 1)

# Define a batch size
batch_size = 32

# Create a transform if needed (e.g., for image preprocessing)
transform = transforms.Compose([Reshape3DTransform(), Uint8ToFloatTransform()])  # Adjust as needed

# Create instances of the custom dataset
custom_dataset = CustomDataset(lbl_data_folder='lbl_data', img_data_folder='img_data', transform=transform)

# Split the dataset into train and validation subsets
train_indices, val_indices = train_test_split(list(range(len(custom_dataset))), test_size=0.2, random_state=42)

# Create DataLoader for training and validation subsets
train_dataset = Subset(custom_dataset, train_indices)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Subset(custom_dataset, val_indices)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Create an instance of the model
model = BallDetectionCNN()

# Define loss function and optimizer
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model in training mode
    
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()  # Zero the gradient buffers
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print training loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}")

    # Validation
    model.eval()  # Set the model in evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Print validation loss and accuracy for this epoch
    print(f"Validation Loss: {val_loss / len(val_dataloader)}")

print("Training finished!")
