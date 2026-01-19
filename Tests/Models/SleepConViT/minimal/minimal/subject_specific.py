class config:
    AMP = True
    BATCH_SIZE_TRAIN = 2*256
    BATCH_SIZE_VALID = 2*256
    EPOCHS = 5000
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e7
    NUM_WORKERS = 4
    SEED = 20
    WEIGHT_DECAY = 0.01
    MODEL_NAME = 'EpiDeNet'
    LEARNING_RATE = 0.000001

import argparse
from utils.models_def import LitEpiDeNet
from pytorch_lightning import Trainer
import os
from pytorch_lightning.loggers import CSVLogger
import re
from utils.dataset import EOGDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from pytorch_lightning import seed_everything
import sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_next_version_number(base_path, experiment_name):
    regex = re.compile(r"version_(\d+)")
    existing_versions = []
    path = os.path.join(base_path, experiment_name)
    if not os.path.exists(path):
        os.makedirs(path)
    for dir_name in os.listdir(path):
        match = regex.match(dir_name)
        if match:
            existing_versions.append(int(match.group(1)))
    return f"version_{max(existing_versions) + 1 if existing_versions else 0}"

if __name__ == '__main__':
    seed_everything(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EpiDeNet', help='Model to use')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--subject', type=int, default=1, help='Subject number')
    config = config()
    args = parser.parse_args()
    config.EPOCHS = args.epochs
    config.MODEL_NAME = args.model
    classes = 11

    # Set path based on your environment
    path_add = '/scratch/thoriri/'

    # Now let's do the Subject Specific Training.
    # We will load the data of subject 1 and train the model on only that data.
    subject = args.subject
    if subject == 0:
        # Here we are doing the global version of the model
        data_path = path_add + 'EOG_data/all_data.npy'
        labels_path = path_add + 'EOG_data/all_labels.npy'
        print("Using all data")
    # add a 0 to the subject number if it is less than 10
    elif subject < 6 and subject > -1:
        subject = f'0{subject}'
        data_path = path_add + f'EOG_data/subject_specific/sub{subject}_data.npy'
        labels_path = path_add + f'EOG_data/subject_specific/sub{subject}_labels.npy'
    else:
        print("Subject number should be less than 6. Exiting.")
        sys.exit(0)


    # For each subject, we will do a 5-fold cross-validation
    # And we will go from 300 samples to 1000 samples in steps of 100 samples
    num_samples_range = range(300, 1001, 100)
    for num_samples in num_samples_range:
        dataset = EOGDataset(data_path, labels_path, num_samples=num_samples)
        kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE_VALID, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
            model = LitEpiDeNet(config)
            if subject == 0:
                logger_name = f'{config.MODEL_NAME}_all_subjects_{str(num_samples)}_fold_{fold}'
            else:
                logger_name = f'{config.MODEL_NAME}_subject_{subject}_{str(num_samples)}_fold_{fold}'

            version = get_next_version_number('lightning_logs', logger_name)
            logger = CSVLogger(
                save_dir='lightning_logs',
                name=logger_name,
                version=version  # Dynamically set the version here
            )

            model = LitEpiDeNet(config=config,output_classes=classes, steps_per_epoch=len(train_loader))

            # Setup Trainer with the fold-specific logger
            trainer = Trainer(
                max_epochs=config.EPOCHS,
                accelerator="gpu", devices=1,
                deterministic=False,
                precision=16 if config.AMP else 32,
                logger=logger,
            )

            # Train
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            print("Fold ", fold, " done")
