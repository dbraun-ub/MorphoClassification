import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from datasets import ThreeViewsDataset, FrontViewDataset, FrontViewDatasetV2 
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

def validation_step(model, criterion, val_loader, device):
    model.eval()
    model.to(device)
    if opt.device == "xpu":
        model = ipex.optimize(model, dtype=torch.float32)
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if opt.linear_target:
                sigmoid = nn.Sigmoid()
                outputs = sigmoid(outputs).squeeze(1)
                loss = criterion(outputs, targets / (opt.num_classes - 1))
                predicted = torch.round(outputs * (opt.num_classes - 1))
            else:
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)

            val_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy

def merge_df_with_markers(df_morpho):
    filename = os.path.join(opt.split_path, 'coordinates_Batch1.txt')
    df_marqueurs = pd.read_csv(filename)
    df2 = df_marqueurs[['patient_id', 'marker', 'x_pix', 'y_pix']]
    df2 = df2[~df2['marker'].str.startswith('SV')]
    df2 = df2[~df2['marker'].str.startswith('FAC')]
    df2 = df2[~df2['marker'].str.startswith('SDC')]
    df2 = df2[~df2['marker'].str.startswith('FPC')]

    # Drop duplicates based on the new column
    df2['patient_marker'] = df2['patient_id'].astype(str) + '_' + df2['marker']
    df2 = df2.drop_duplicates(subset='patient_marker', keep='first').reset_index(drop=True)

    # Drop the patient_marker column if it's no longer needed
    df2 = df2.drop(columns=['patient_marker'])

    df_pivot = df2.pivot(index='patient_id', columns='marker', values=['x_pix', 'y_pix'])
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    df_pivot.reset_index(inplace=True)

    df = pd.merge(df_morpho, df_pivot, on='patient_id', how='left')

    return df



def train(opt):
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cudnn's auto-tuner for reproducible results

    # device = torch.device("cpu")
    if opt.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("cuda is not available")
            device = torch.device("cpu")
    elif opt.device == "xpu":
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            device = "xpu"
        else:
            opt.device = "cpu"
            print("xpu is not available")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    
    ## Variables
    print('-'*20)
    print(opt)
    print('-'*20)



    ## Define your dataset
    list_datasets = {
        'FrontViewDataset': FrontViewDataset,
        'ThreeViewsDataset': ThreeViewsDataset,
        'FrontViewDatasetV2': FrontViewDatasetV2,
    }   

    # image name list
    train_df = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in opt.train_folds], axis=0)
    val_df   = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in opt.val_folds], axis=0)

    # Init datasets
    MyDataset = list_datasets[opt.dataset]
    in_channels, height, width = opt.image_size
    if opt.dataset == 'FrontViewDatasetV2':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_file_list = merge_df_with_markers(train_df)
        val_file_list = merge_df_with_markers(val_df)
    else:
        transform = create_transform((in_channels,height,width))
        train_file_list = train_df.values
        val_file_list = val_df.values

    train_dataset = MyDataset(train_file_list, opt.data_path, transform, num_classes=opt.num_classes)
    val_dataset   = MyDataset(val_file_list, opt.data_path, transform, num_classes=opt.num_classes)

    # Define data loader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

    ## Create model using timm
    # print("model not loaded yet")
    if opt.linear_target:
        model = timm.create_model(opt.model_name, pretrained=True, in_chans=in_channels, num_classes=1, drop_rate=opt.drop_rate)
    else:
        model = timm.create_model(opt.model_name, pretrained=True, in_chans=in_channels, num_classes=opt.num_classes, drop_rate=opt.drop_rate)
    # print("model loaded")

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
    if opt.linear_target:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = StepLR(optimizer, step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(opt.log_path, opt.log_name))

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=opt.earlyStopping_patience, min_delta=opt.earlyStopping_min_delta)

    
    unfreeze_step = 0
    unfreeze_progression = {
        'convnext_tiny': {
            0: model.stages[3],
            1: model.stages[2],
            2: model.stages[1],
            3: model.stages[0],
            4: model.stem,
        }
    }

    # device = set_device(opt)

    # Training loop
    best_val_acc = 0
    # best_val_loss = 1e8 # almost like inf value
    for epoch in range(opt.num_epochs):
        model.train()
        model.to(device)
        if opt.device == "xpu":
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)

        if opt.progressive_unfreeze:
            if epoch >= opt.num_epoch_unfreeze:
                for param in unfreeze_progression[opt.model_name][unfreeze_step].parameters():
                    param.requires_grad = True
                unfreeze_step += 1
                if unfreeze_step > len(unfreeze_progression[opt.model_name]):
                    opt.progressive_unfreeze = False
        else: 
            if epoch == opt.num_epoch_unfreeze:
                for param in model.parameters():
                    param.requires_grad = True

        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if opt.linear_target:
                sigmoid = nn.Sigmoid()
                outputs = sigmoid(outputs).squeeze(1)
                loss = criterion(outputs, targets / (opt.num_classes - 1))
                predicted = torch.round(outputs * (opt.num_classes - 1))
            else:
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
            loss.backward()
            optimizer.step()

            # for logging
            running_loss += loss.item()

            # _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch+1}/{opt.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}, Acc: {100 * (predicted == targets).sum().item() / targets.size(0)}')

            if (epoch == 0) and batch_idx == 0:
                img_grid = vutils.make_grid(inputs.cpu(), normalize=True)
                writer.add_image('Images/train', img_grid, epoch)

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        train_accuracy = 100 * correct / total
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.3f}%')
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Step the scheduler
        scheduler.step()



        # Validation step
        val_loss, val_accuracy = validation_step(model, criterion, val_loader, device)

        print(f'Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_accuracy:.3f}%')

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # if (epoch > 3) and (val_accuracy > best_val_acc):
        #     best_val_acc = val_accuracy
        #     torch.save(model.state_dict(), f'runs/{opt.log_name}/model_epoch.pth') # Don't specify the epoch number. Simply save the best result

        # Generally performs worse or equal
        # # Save the model if the validation loss has decreased
        # if (val_loss < best_val_loss):
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), f'runs/{opt.log_name}/model_best_loss.pth')

        
        # Save the model if the validation accuracy has increased
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), f'runs/{opt.log_name}/model_best_acc.pth')
        
        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    writer.close()


if __name__ == '__main__':
    opt = options()
    train(opt)
