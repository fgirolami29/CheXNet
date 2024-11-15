# chexnet_train.py
# Set the number of intra-op and inter-op threads to manage CPU utilization
from tkinter import Image
import torch
#torch.set_num_threads(4)
#torch.set_num_interop_threads(4)

import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from chexnet_main_model import DenseNet121, ChestXrayDataSet  # Replace 'main_model_file' with the actual filename

# Parameters
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images0'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
CKPT_PATH = 'models/chexS/checkpoint/checkpoint.pth.tar'
def __getitem__(self, index):
    image_name = os.path.join(self.data_dir, self.image_list[index])
    try:
        image = Image.open(image_name).convert('RGB')
    except FileNotFoundError:
        print(f"Missing file: {image_name}. Skipping.")
        return None

    if self.transform is not None:
        image = self.transform(image)

    label = self.labels[index]
    return image, label
def train():
    # Initialize the model
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model)
    model.train()

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])(crop) for crop in crops]))
    ])

    # Load training data
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            if inputs is None:  # Skip invalid batches
                continue

            # Flatten crops dimension
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)  # Flatten crops: [BATCH_SIZE * 10, 3, 224, 224]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # Shape: [BATCH_SIZE * 10, N_CLASSES]
            outputs = outputs.view(bs, ncrops, -1).mean(dim=1)  # Average crops: [BATCH_SIZE, N_CLASSES]

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

        # Save checkpoint
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, CKPT_PATH)
        print(f"Checkpoint saved for epoch {epoch+1}")

    print("Training complete")

def trainNO():
    # Initialize the model
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model)
    model.train()

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])(crop) for crop in crops]))
    ])

    # Load training data
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE // 10, shuffle=True, num_workers=0, pin_memory=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            if inputs is None:  # Skip invalid batches
                continue

            # Flatten crops dimension
            inputs = inputs.view(-1, 3, 224, 224)  # [BATCH_SIZE * 10, 3, 224, 224]
            targets = targets.repeat_interleave(10, dim=0)  # Repeat targets for crops

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # [BATCH_SIZE * 10, N_CLASSES]
            outputs = outputs.view(-1, 10, N_CLASSES).mean(dim=1)  # Average crops: [BATCH_SIZE, N_CLASSES]

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

        # Save checkpoint
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, CKPT_PATH)
        print(f"Checkpoint saved for epoch {epoch+1}")

    print("Training complete")

if __name__ == '__main__':
    train()
