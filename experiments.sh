clear;
python train.py --log_name resnet50_004 --data_path /home/ubuntu/data/MORPHO_Batch1 --model_name resnet50 --num_epoch_unfreeze 10 --train_folds 0 1 2 3 4 5 6 7 --val_folds 8 --test_folds 9

clear;
python train.py --log_name resnet50_005 --data_path /home/ubuntu/data/MORPHO_Batch1 --model_name resnet50 --num_epoch_unfreeze 10 --train_folds 1 2 3 4 5 6 7 9 --val_folds 8 --test_folds 0

clear;
python train.py --log_name resnet50_006 --data_path /home/ubuntu/data/MORPHO_Batch1 --model_name resnet50 --num_epoch_unfreeze 10 --train_folds 0 2 4 5 6 7 8 9 --val_folds 8 --test_folds 1