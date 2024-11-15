# chexnet_train.py
# Set the number of intra-op and inter-op threads to manage CPU utilization
import torch
#torch.set_num_threads(4)
#torch.set_num_interop_threads(4)

import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from main_model import DenseNet121, ChestXrayDataSet  # Replace 'main_model_file' with the actual filename

# Parameters
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
CKPT_PATH = 'models/chexS/checkpoint/checkpoint.pth.tar'

def train():
    # Initialize the model
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model)
    model.train()  # Set model to training mode

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

 # Define the transformation functions
    def stack_tensors(crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])
    def normalize_crops(crops):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return torch.stack([normalize(crop) for crop in crops])

                                    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(stack_tensors),
        transforms.Lambda(normalize_crops)
    ])

    # Load training data
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)



    # Training loop
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.view(-1, 3, 224, 224), targets  # Reshape input for TenCrop
            optimizer.zero_grad()  # Zero gradients
            # Forward pass
            outputs = model(inputs)  # Shape: [BATCH_SIZE * 10, N_CLASSES]
            outputs = outputs.view(BATCH_SIZE, 10, N_CLASSES)  # Reshape to [BATCH_SIZE, 10, N_CLASSES]
            outputs = outputs.mean(dim=1)  # Average across the 10 crops to get [BATCH_SIZE, N_CLASSES]
            
            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # Save checkpoint at the end of each epoch
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, CKPT_PATH)
        print(f"Checkpoint saved for epoch {epoch+1}")

    print("Training complete")

if __name__ == '__main__':
    train()
