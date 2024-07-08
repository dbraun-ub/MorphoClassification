#!/bin/bash

# Define the configurations in an array
declare -a configs=(
    "mobilenetv2_120d_002 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name mobilenetv2_120d --num_epoch_unfreeze 10 --earlyStopping_patience 5"
    "resnet50d_002 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    "resnet50d_003 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    "resnet50d_004 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    "resnet50d_005 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    "resnet50d_006 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 5 7 8 9 --val_folds 0 --test_folds 6 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
)

# Loop over each configuration
for config in "${configs[@]}"
do
    # clear

    # Extract log_name from config
    log_name=$(echo $config | cut -d ' ' -f 1)

    # Run Python script
    echo python train.py --log_name ${config}
    python train.py --log_name ${config}

    # Zip and clean up
    cd runs
    zip -r ${log_name}.zip $log_name
    rm -r $log_name
    cd ..
    
    # clear
done



# clear;
# python train.py --log_name resnet50_004 --data_path /home/ubuntu/data/MORPHO_Batch1 --model_name resnet50 --num_epoch_unfreeze 10 --train_folds 0 1 2 3 4 5 6 7 --val_folds 8 --test_folds 9
# cd runs
# zip -r resnet50_004.zip resnet50_004
# rm -r resnet50_004
# cd ..

# clear;
# python train.py --log_name resnet50_005 --data_path /home/ubuntu/data/MORPHO_Batch1 --model_name resnet50 --num_epoch_unfreeze 10 --train_folds 1 2 3 4 5 6 7 9 --val_folds 8 --test_folds 0
# cd runs
# zip -r resnet50_005.zip resnet50_005
# rm -r resnet50_005
# cd ..

# clear;
# python train.py --log_name resnet50_006 --data_path /home/ubuntu/data/MORPHO_Batch1 --model_name resnet50 --num_epoch_unfreeze 10 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1
# cd runs
# zip -r resnet50_006.zip resnet50_006
# rm -r resnet50_006
# cd ..

