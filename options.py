import argparse
import json
import os

def options():
    parser = argparse.ArgumentParser(description="Training a deep learning model with custom parameters.")
    
    # Model
    parser.add_argument('--model_name', type=str, default='resnet50', help='Name of the model to use from timm')

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--train_folds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7] ,help='Sizes of the layers in the model')
    parser.add_argument('--val_folds', type=int, nargs='+', default=[8] ,help='Sizes of the layers in the model')
    parser.add_argument('--test_folds', type=int, nargs='+', default=[9] ,help='Sizes of the layers in the model')
    parser.add_argument('--split', type=str, default='full_balanced', help='Split to use for training')
    parser.add_argument('--dataset', type=str, default='FrontViewDataset', help='Dataset to use for training')
    parser.add_argument('--image_size', type=int, nargs='+', default=[3,320,224], help='Input image size')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop rate')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='Scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Scheduler gamma value')
    parser.add_argument('--num_epoch_unfreeze', type=int, default=100, help='Epoch number when to unfreeze all parameters from the network')
    parser.add_argument('--progressive_unfreeze', action='store_true', help='Activate or deactivate the progressive unfreeze')
    parser.add_argument('--progressive_unfreeze_step', type=int, default=1, help='unfreeze step in the case the progressive unfreeze is activated')
    parser.add_argument('--earlyStopping_min_delta', type=float, default=0.01, help='EarlyStopping delta margin')
    parser.add_argument('--earlyStopping_patience', type=int, default=5, help='Number of epochs without improvement in the validation loss before stopping')
    parser.add_argument('--view', type=str, choices=['FA', 'SD', 'FP', 'all'], default='FA', help='Split to use for training')
    parser.add_argument('--transform', action='store_true', help='Activate or deactivate the progressive unfreeze')
    parser.add_argument('--weighted_loss', action='store_true', help='Apply a weight on the loss to balance the classes')
    parser.add_argument('--weight_index', action='store_true', help='Apply a weight on the loss to balance the classes')
    # Loss
    parser.add_argument('--linear_target', action='store_true', help='Evaluate a linear value instead of classes')
    
    # device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
 
    # Path
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--split_path', type=str, default='./splits', help='Path to the split')
    parser.add_argument('--log_path', type=str, default='./runs', help='Path to log runs')

    # Evaluation

    # Log
    parser.add_argument('--log_name', type=str, help='Name of the log run')

    opt = parser.parse_args()
    save_options(opt)
    return opt

def save_options(opt, filename='options.json'):
    file_path = os.path.join(opt.log_path, opt.log_name, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(vars(opt), f, indent=4)

def load_options(filename='options.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            args_dict = json.load(f)
        return argparse.Namespace(**args_dict)
    else:
        return None


if __name__ == '__main__':
    options()
 
