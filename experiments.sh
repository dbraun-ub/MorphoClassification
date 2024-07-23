#!/bin/bash

# Define the configurations in an array
declare -a configs=(
    # "mobilenetv2_120d_002 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name mobilenetv2_120d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    # "resnet50d_002 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    # "resnet50d_003 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    # "resnet50d_004 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    # "resnet50d_005 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    # "resnet50d_006 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 5 7 8 9 --val_folds 0 --test_folds 6 --model_name resnet50d --num_epoch_unfreeze 100 --earlyStopping_patience 5"
    # "resnet50d_007_bis --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10"
    # "resnet50d_007_ter --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10"
    # "resnet50d_008 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10"
    # "resnet50d_009 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10"
    # "resnet50d_010 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10"
    # "resnet50d_011 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10"
    # "resnet50d_012 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 5 7 8 9 --val_folds 0 --test_folds 6 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10"
    # "resnet50d_013 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_014 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_015 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_016 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_017 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_018 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 5 7 8 9 --val_folds 0 --test_folds 6 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_019 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_020 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_021 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --drop_rate 0.1 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_022 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --drop_rate 0.1 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_023 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --drop_rate 0.1 --dataset ThreeViewsDataset --image_size 3 224 320"
    # "resnet50d_024 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --drop_rate 0.1 --dataset ThreeViewsDataset --image_size 3 224 320"
    
    # "resnet50d_025 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --dataset FrontViewDataset --image_size 3 320 224"
    # "resnet50d_026 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --dataset FrontViewDataset --image_size 3 320 224"
    # "resnet50d_027 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --drop_rate 0.1 --dataset FrontViewDataset --image_size 3 320 224"
    # "resnet50d_028 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --drop_rate 0.1 --dataset FrontViewDataset --image_size 3 320 224"
    # "resnet50d_029 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --drop_rate 0.1 --dataset FrontViewDataset --image_size 3 320 224"
    # "resnet50d_030 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 5 --earlyStopping_min_delta 0 --learning_rate 0.0001 --drop_rate 0.1 --dataset FrontViewDataset --image_size 3 320 224"

    # "resnet50d_031 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 5"
    # "resnet50d_032 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 5"
    # "resnet50d_033 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 5"
    # "resnet50d_034 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 5"
    # "resnet50d_035 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 5"
 
    # "resnet50d_036 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDataset --image_size 3 320 224 --scheduler_step_size 5"
    # "resnet50d_037 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDataset --image_size 3 320 224 --scheduler_step_size 5"
    # "resnet50d_038 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDataset --image_size 3 320 224 --scheduler_step_size 5"
    # "resnet50d_039 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDataset --image_size 3 320 224 --scheduler_step_size 5"
    # "resnet50d_040 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name resnet50d --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDataset --image_size 3 320 224 --scheduler_step_size 5"

    # "convnext_tiny_002 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
    # "convnext_tiny_003 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
    # "convnext_tiny_004 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
    
    # "resnet33ts_002 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name resnet33ts --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
    # "resnet33ts_003 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name resnet33ts --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
    # "resnet33ts_004 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name resnet33ts --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
    # "resnet33ts_005 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name resnet33ts --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
    # "resnet33ts_006 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name resnet33ts --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0"
  
    # "convnext_tiny_005 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 5 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 5"
    # "convnext_tiny_006 --data_path /data --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 5 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 5"
    
    # "convnext_tiny_007 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 5 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 5"
    # "convnext_tiny_008 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name convnext_tiny --num_epoch_unfreeze 5 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 5"
    # "convnext_tiny_009 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name convnext_tiny --num_epoch_unfreeze 5 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 5"

    # "convnext_tiny_005 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 100 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 2 --num_epoch_unfreeze 3"
    # "convnext_tiny_006 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 100 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 2 --num_epoch_unfreeze 3"
    # "convnext_tiny_007 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 100 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 2 --num_epoch_unfreeze 3"
    # "convnext_tiny_008 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 2 3 5 6 7 8 --val_folds 9 --test_folds 4 --model_name convnext_tiny --num_epoch_unfreeze 100 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 2 --num_epoch_unfreeze 3"
    # "convnext_tiny_009 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 1 3 4 5 6 7 9 --val_folds 2 --test_folds 8 --model_name convnext_tiny --num_epoch_unfreeze 100 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --progressive_unfreeze --progressive_unfreeze_step 2 --num_epoch_unfreeze 3"

    # "convnext_tiny_010 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.0001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --device cuda"
    # "convnext_tiny_011 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.0001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --device cuda"
    # "convnext_tiny_012 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.0001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --device cuda"

    # "convnext_tiny_013 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.0001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --batch_size 64 --device cuda"
    # "convnext_tiny_014 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.0001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --batch_size 64 --device cuda"
   
    # "convnext_tiny_015 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --earlyStopping_min_delta 0 --learning_rate 0.0001 --dataset FrontViewDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --batch_size 64 --device cuda"
    # "convnext_tiny_016 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --earlyStopping_min_delta 0 --learning_rate 0.0001 --dataset FrontViewDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --batch_size 64 --device cuda"
    # "convnext_tiny_017 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --earlyStopping_min_delta 0 --learning_rate 0.0001 --dataset FrontViewDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --linear_target --batch_size 64 --device cuda"

    # "convnext_tiny_018 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --num_classes 3 --split full_balanced_3_classes"
    # "convnext_tiny_019 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --num_classes 3 --split full_balanced_3_classes"
    # "convnext_tiny_020 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDataset --image_size 3 224 320 --scheduler_step_size 10 --drop_rate 0 --num_classes 3 --split full_balanced_3_classes"
    
    # "convnext_tiny_021 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDatasetV2 --image_size 3 240 640 --scheduler_step_size 10 --drop_rate 0"
    # "convnext_tiny_022 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDatasetV2 --image_size 3 240 640 --scheduler_step_size 10 --drop_rate 0"
    
    "convnext_tiny_024 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDatasetV2 --image_size 3 512 512 --scheduler_step_size 10 --drop_rate 0"
    "convnext_tiny_023 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset FrontViewDatasetV2 --image_size 3 240 640 --scheduler_step_size 10 --drop_rate 0"
    "convnext_tiny_025 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDatasetV2 --image_size 3 512 512 --scheduler_step_size 10 --drop_rate 0"
    "convnext_tiny_026 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDatasetV2 --image_size 3 512 512 --scheduler_step_size 10 --drop_rate 0"

    "convnext_tiny_027 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 3 4 5 6 7 9 --val_folds 8 --test_folds 1 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDatasetV2 --image_size 3 224 224 --scheduler_step_size 10 --drop_rate 0"
    "convnext_tiny_028 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 0 2 4 5 6 7 8 9 --val_folds 1 --test_folds 3 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDatasetV2 --image_size 3 224 224 --scheduler_step_size 10 --drop_rate 0"
    "convnext_tiny_029 --data_path /home/ubuntu/data/MORPHO_Batch1 --train_folds 1 2 3 4 6 7 8 9 --val_folds 5 --test_folds 0 --model_name convnext_tiny --num_epoch_unfreeze 10 --earlyStopping_patience 10 --learning_rate 0.001 --dataset ThreeViewsDatasetV2 --image_size 3 224 224 --scheduler_step_size 10 --drop_rate 0"

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

