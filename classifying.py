import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetDataset, collate_fn_gpt
from model import Video2Spot, Video2Classifcation
from train import trainer, test_spotting
from loss import NLLLoss

def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetDataset(path=args.SoccerNet_path, pos_path=args.Position_path, 
                                        features=args.features, split=args.split_train, version=args.version, 
                                        framerate=args.framerate, caption_window_size=args.window_size_caption, 
                                        spot_window_size=args.window_size_spotting, insert_empty_caption=False)
        dataset_Valid = SoccerNetDataset(path=args.SoccerNet_path, pos_path=args.Position_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, 
                                     caption_window_size=args.window_size_caption, spot_window_size=args.window_size_spotting, insert_empty_caption=False)
        dataset_Valid_metric  = SoccerNetDataset(path=args.SoccerNet_path, pos_path=args.Position_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, 
                                     caption_window_size=args.window_size_caption, spot_window_size=args.window_size_spotting, insert_empty_caption=False)
    dataset_Test = SoccerNetDataset(path=args.SoccerNet_path, pos_path=args.Position_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, 
                                     caption_window_size=args.window_size_caption, spot_window_size=args.window_size_spotting, insert_empty_caption=False)

    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][1].shape[-1]
        print("feature_dim found:", args.feature_dim)
    # create model
    feature_size = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}
    model = Video2Classifcation(num_classes=len(dataset_Test.class_labels), weights=args.load_weights, input_size=args.feature_dim,
                  window_size=args.window_size_spotting, 
                  framerate=args.framerate, pool=args.pool, freeze_encoder=args.freeze_encoder, weights_encoder=args.weights_encoder,
                  proj_size=feature_size[args.gpt_type]).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    # parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_gpt)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_gpt)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_gpt)


    # training parameters
    if not args.test_only:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # start training
        trainer("classifying", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=10, evaluation_frequency=10, wandb_use=args.wandb, debug=args.debug)

    return None
