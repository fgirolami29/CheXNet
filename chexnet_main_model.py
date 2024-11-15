# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
#import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score

from rich import print



#from colorama import Fore, Style, init
#init()  # Initialize colorama for Windows compatibility

CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
#DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 64


def main():

    #cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model)

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        #model.load_state_dict(checkpoint['state_dict'])
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove `module.` if it exists
            new_state_dict[name] = v
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    #normalize = transforms.Normalize([0.485, 0.456, 0.406],
     #                                [0.229, 0.224, 0.225])


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


    # Initialize dataset and dataloader
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=TEST_IMAGE_LIST, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        #target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        #input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w))
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    # fix nan AUROC_avg = np.array(AUROCs).mean()
    AUROC_avg = np.nanmean(AUROCs)  # Replaces np.mean with np.nanmean to ignore nan values


    # Print AUROC for each class, displaying "SKIPPED" for NaN values
    for i in range(N_CLASSES):
        if np.isnan(AUROCs[i]):
            #print('The AUROC of {} is SKIPPED'.format(CLASS_NAMES[i]))
            print(f"[italic dark_orange strike]The AUROC of {CLASS_NAMES[i]} is SKIPPED[/italic dark_orange strike]")
        else:
            #print('The AUROC of {} is {:.3f}'.format(CLASS_NAMES[i], AUROCs[i]))
            print(f"[underline green]The AUROC of {CLASS_NAMES[i]} is [yellow]{AUROCs[i]:.3f}[/yellow][/underline green]")


    #print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

    # Print average AUROC with conditional formatting
    if AUROC_avg > 0:
        print(f"[bold green]The average AUROC is [yellow]{AUROC_avg:.3f}[/yellow][/bold green]")
    else:
        print(f"[bold red]The average AUROC is {AUROC_avg:.3f}[/bold red]")
    
def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        # Check if only one class is present in y_true for this class
        if len(np.unique(gt_np[:, i])) == 1:
            #print(f"Only one class present in ground truth for {CLASS_NAMES[i]}. Skipping AUC calculation.")
            print(f"[italic cyan]Only one class present in ground truth for {CLASS_NAMES[i]}. Skipping AUC calculation.[/italic cyan]")

            AUROCs.append(np.nan)  # You can choose another placeholder if needed
        else:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x



if __name__ == '__main__':
    main()