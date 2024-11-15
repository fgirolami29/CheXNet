# chexnet_checkpoint.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
from chexnet_main_model import DenseNet121, ChestXrayDataSet  # Replace 'main_model_file' with the actual filename

# Parameters
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
CKPT_PATH = 'models/chexS/checkpoint/checkpoint.pth.tar'

def load_or_initialize_model():
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model)

    if os.path.isfile(CKPT_PATH):
        print("=> Loading checkpoint...")
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Checkpoint loaded successfully")
    else:
        print("=> No checkpoint found. Initializing model from scratch.")
    return model

def train(model):
    # Set the model to training mode
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.view(-1, 3, 224, 224), targets  # Reshape for TenCrop
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(BATCH_SIZE, N_CLASSES)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # Save checkpoint after each epoch
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, CKPT_PATH)
        print(f"Checkpoint saved for epoch {epoch+1}")

    print("Training complete")

def evaluate(model):
    # Set the model to evaluation mode
    model.eval()

    # Define transformations for test dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])(crop) for crop in crops]))
    ])

    # Load test data
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=TEST_IMAGE_LIST, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.view(-1, 3, 224, 224)
            outputs = model(inputs)
            outputs_mean = outputs.view(BATCH_SIZE, N_CLASSES).mean(1)

            gt = torch.cat((gt, targets), 0)
            pred = torch.cat((pred, outputs_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.nanmean(AUROCs)
    print(f"The average AUROC is {AUROC_avg:.3f}")

    for i in range(N_CLASSES):
        if np.isnan(AUROCs[i]):
            print(f"The AUROC of {CLASS_NAMES[i]} is SKIPPED")
        else:
            print(f"The AUROC of {CLASS_NAMES[i]} is {AUROCs[i]:.3f}")

def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        if len(np.unique(gt_np[:, i])) == 1:
            AUROCs.append(np.nan)
        else:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

if __name__ == '__main__':
    model = load_or_initialize_model()
    if not os.path.isfile(CKPT_PATH):
        train(model)
    evaluate(model)
