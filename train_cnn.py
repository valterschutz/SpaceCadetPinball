import glob
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

# Dimensions of pinball board
WIDTH = 600;
HEIGHT = 416;

SAVED_WIDTH = WIDTH // 2
SAVED_HEIGHT = HEIGHT // 2

def get_device():
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

device = get_device()

class BallDetectionCNN(nn.Module):
    def __init__(self):
        super(BallDetectionCNN, self).__init__()
        
        lin_size = int(SAVED_WIDTH/(2**6)) * int(SAVED_HEIGHT/(2**6)) * 16

        self.conv = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Flatten(),

            nn.Linear(lin_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
        )


    def forward(self, x):
        x = self.conv(x)
        # print(f"after conv: {x.shape}")
        x = self.lin(x)
        # print(f"after lin: {x.shape}")
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

        lbl_data = torch.zeros(128)
        lbl_data[:8] = torch.load(lbl_filename).reshape(4)
        img_data = torch.load(img_filename)

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, lbl_data

class Uint8ToFloatTransform(object):
    def __call__(self, sample):
        return sample.float() / 255.0

class Reshape3DTransform(object):
    def __call__(self, sample):
        return sample.view(
            sample.shape[0]*sample.shape[1],
            sample.shape[2],
            sample.shape[3]
        )

class CustomMSELoss(nn.Module):
    def __init__(self, weights):
        super(CustomMSELoss, self).__init__()
        self.weights = weights.to(device)

    def forward(self, output, target):
        thing0 =  (output - target)**2
        thing1 = self.weights * thing0
        loss = torch.sum(thing1)
        return loss

def print_model_summary(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

def train_model(model, num_epochs, batch_size, lr):
    print_model_summary(model)
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

    # Define loss function and optimizer
    # criterion = nn.HuberLoss()
    weights = 1e-5 * torch.ones(128)
    weights[:8] = 1
    weights = weights / weights.sum()
    weights = weights.to(device)
    # criterion = CustomMSELoss(weights)
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("Training started!")
    for epoch in range(num_epochs):
        model.train()  # Set the model in training mode
        
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradient buffers
            
            # Forward pass
            outputs = model(inputs)
            
            # print(f"size of outputs: {outputs.shape}")
            # print(f"size of labels: {labels.shape}")
            # Compute loss
            # loss = criterion(weights*outputs, weights*labels)
            loss = criterion(outputs, labels)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print training loss for this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataloader):.5f}")

        # Validation
        model.eval()  # Set the model in evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(weights*outputs, weights*labels)
                val_loss += loss.item()

        # Print validation loss and accuracy for this epoch
        print(f"Validation Loss: {val_loss / len(val_dataloader):.5f}")

    print("Training finished!")
    return model

def load_latest_model():
    # Define the directory where your model files are stored
    model_directory = 'models'  # Replace with the actual directory path

    # List all model files in the directory
    model_files = glob.glob(os.path.join(model_directory, 'model_*.pth'))

    # Ensure that there are model files to load
    if not model_files:
        return None

    # Sort the model files by timestamp (modification time) in descending order
    model_files.sort(key=os.path.getmtime, reverse=True)

    # Select the most recent model file
    latest_model_path = model_files[0]

    print(f"Loading {latest_model_path}...")

    model = BallDetectionCNN()
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    
    # Load to GPU
    model.to(device)
    return model

def save_model(model):
    # Get the current date and time as a string
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Define a file path with the timestamp
    model_path = f'models/model_{current_datetime}.pth'
    print(f"Saving {model_path}...")
    torch.save(model.state_dict(), model_path)

def plot_predictions(model):
    transform = transforms.Compose([Reshape3DTransform(), Uint8ToFloatTransform()])  # Adjust as needed

    # Create instances of the custom dataset
    custom_dataset = CustomDataset(lbl_data_folder='lbl_data', img_data_folder='img_data', transform=transform)

    dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            labels = labels[0,:2].cpu()
            # inputs = inputs[0,:3,:,:]
            outputs = outputs[0,:2].cpu()

            plt.plot([-labels[0], -outputs[0]], [-labels[1], -outputs[1]], alpha=0.5)

    plt.savefig("figs/latest_fig.png")
    plt.show()

            # loss = criterion(outputs, labels)
            # val_loss += loss.item()




def train(num_epochs, batch_size, lr):
    # Load latest model
    model = load_latest_model()
    if model == None:
        print("Creating new model...")
        model = BallDetectionCNN()
        model.to(device)

    # Train it
    trained_model = train_model(model, num_epochs, batch_size, lr)

    # Save it
    save_model(trained_model)


def main():
    if sys.argv[1] == 'train':
        num_epochs = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        lr = float(sys.argv[4])
        train(num_epochs, batch_size, lr)
    elif sys.argv[1] == 'plot':
        plot_predictions(load_latest_model())
    else:
        error('hej')

if __name__ == "__main__":
    main()
