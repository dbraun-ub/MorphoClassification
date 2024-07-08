import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from datasets import ThreeViewsDataset, FrontViewDataset  # Replace with your actual dataset class
import timm
import pandas as pd
import numpy as np

from timm.data.transforms_factory import create_transform
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
from options import options

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(opt):
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cudnn's auto-tuner for reproducible results

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## Variables
    # print(f'model name: {opt.model_name}')
    # print(f'log name: {opt.log_name}')
    # print(f'split: {opt.split}')
    # print(f'image size: {opt.image_size}')
    # print(f'num classes: {opt.num_classes}')
    # print(f'dataset: {opt.dataset}')
    # print(f'dataset path: {opt.dataset_path}')
    # print(f'train folds: {opt.train_folds}')
    # print(f'val folds: {opt.val_folds}')
    # print(f'batch size: {opt.batch_size}')
    # print(f'learning rate: {opt.learning_rate}')
    # print(f'num epochs max: {opt.num_epochs}')
    # print(f'num epochs max: {opt.num_epochs}')
    print('-'*20)
    print(opt)
    print('-'*20)



    # split = opt.split
    # split_dir = opt.split_path#r'C:\Users\Daniel\Documents\Workspace\ClinicDataAnalysis\splits'
    # image_dir = opt.dataset_path#r'C:\Users\Daniel\Documents\Data\Batch1'
    # train_folds = opt.train_folds#[0,1,2,3,4,5,6,7]
    # val_folds = opt.val_folds#[8]
    # test_folds = [9] 
    # num_classes = opt.num_classes#5
    # in_channels = 3
    # log_name = opt.log_name#model_name + '_' + '001'


    ## Define your dataset
    list_datasets = {
        'FrontViewDataset': FrontViewDataset,
        'ThreeViewsDataset': ThreeViewsDataset
    }   

    # image name list
    train_df = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in opt.train_folds], axis=0)
    val_df   = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in opt.val_folds], axis=0)

    # Init datasets
    MyDataset = list_datasets[opt.dataset]
    in_channels, height, width = opt.image_size
    transform = create_transform((in_channels,height,width))

    train_dataset = MyDataset(train_df.values, opt.data_path, transform)
    val_dataset   = MyDataset(val_df.values, opt.data_path, transform)

    # Define data loader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    ## Create model using timm
    model = timm.create_model(opt.model_name, pretrained=True, in_chans=in_channels, num_classes=opt.num_classes, drop_rate=opt.drop_rate).to(device)

    # Fine tunning: only train the fc layer
    for param in model.parameters():
        param.requires_grad = False

    if 'resnet50' in opt.model_name:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif 'convnext' in opt.model_name or 'swin' in opt.model_name or opt.model_name == "resnet33ts":
        for param in model.head.parameters():
            param.requires_grad = True
    elif 'levit' in opt.model_name:
        for param in model.head.parameters():
            param.requires_grad = True
        for param in model.head_dist.parameters():
            param.requires_grad = True
    else: # mobilenetv2_120d
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = StepLR(optimizer, step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(opt.log_path, opt.log_name))

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=opt.earlyStopping_patience, min_delta=opt.earlyStopping_min_delta)

    # Training loop
    best_val_acc = 0
    for epoch in range(opt.num_epochs):
        model.train()

        if epoch == opt.num_epoch_unfreeze:
            for param in model.parameters():
                param.requires_grad = True

        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # for logging
            running_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch+1}/{opt.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')

            if (epoch == 0) and batch_idx == 0:
                img_grid = vutils.make_grid(inputs.cpu(), normalize=True)
                writer.add_image('Images/train', img_grid, epoch)

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Step the scheduler
        scheduler.step()

        # Validation step
        # model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if (epoch == 0) and batch_idx == 0:
                    img_grid = vutils.make_grid(inputs.cpu(), normalize=True)
                    writer.add_image('Images/val', img_grid, epoch)
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        if (epoch > 3) and (val_accuracy > best_val_acc):
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), f'runs/{opt.log_name}/model_epoch.pth') # Don't specify the epoch number. Simply save the best result
        
        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    writer.close()


if __name__ == '__main__':
    opt = options()
    train(opt)
