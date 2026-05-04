import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
import os
import torch
import logging
import time
from datetime import datetime

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def valid_probability(value):
    fvalue = float(value)
    if fvalue <= 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid probability between 0 and 1")
    return fvalue


def run_selected_stages(args):
    # Official ECAI 2025 release:
    # only spotting.py and captioning.py are used in the pipeline.
    # The execution order is:
    #   1) spotting
    #   2) captioning
    # captioning.dvc() is kept as the dense-caption inference step inside
    # the captioning module, so the default full pipeline is spotting ->
    # caption -> dvc.
    stage_sequence = ["spotting", "caption", "dvc"] if args.stage == "full" else [args.stage]

    for stage_name in stage_sequence:
        stage_start = time.time()

        if stage_name == "spotting":
            import spotting
            spotting.main(args)
        elif stage_name == "caption":
            import captioning
            captioning.main(args)
        elif stage_name == "dvc":
            import captioning
            args.weights_encoder = None
            captioning.dvc(args)
        else:
            raise ValueError(f"Unsupported stage: {stage_name}")

        logging.info(f"Stage {stage_name} finished in {time.time()-stage_start} seconds")




if __name__ == '__main__':


    parser = ArgumentParser(description='SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="path/to/soccernet",     help='Path for SoccerNet' )
    parser.add_argument('--Position_path',   required=False, type=str,   default="path/to/positions",     help='Path for positions' )
    parser.add_argument('--features',   required=False, type=str,   default="baidu_soccer_embeddings.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=15,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="gpt2-train",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--debug', required=False, action='store_true',  help='Debug mode' )
    parser.add_argument('--wandb', required=False, action='store_true',  help='Use wandb for logging' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test"], help='list of split for testing')

    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim', required=False, type=int,   default=None,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=15,     help='Number of chunks per epoch' )
    parser.add_argument('--framerate', required=False, type=int,   default=1,     help='Framerate of the input features' )
    parser.add_argument('--pool',       required=False, type=str,   default="QFormer", help='How to pool' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )
    parser.add_argument('--min_freq',       required=False, type=int,   default=5, help='Minimum word frequency to the vocabulary for caption generation' )
    parser.add_argument('--teacher_forcing_ratio',  required=False, type=valid_probability,   default=1, help='Teacher forcing ratio to use' )

    parser.add_argument('--first_stage',  required=False, type=str,  choices=["spotting", "caption"], default="spotting")
    parser.add_argument('--window_size_spotting', required=False, type=int,   default=30,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--window_size_caption', required=False, type=int,   default=30,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--freeze_encoder',  required=False, action='store_true',  help='Perform testing only')
    parser.add_argument('--pretrain',   required=False, action='store_true',
                        help='Legacy flag kept for backward compatibility; not used in the official release pipeline' )
    parser.add_argument('--weights_encoder',  required=False, type=str, default=None)
    parser.add_argument('--num_layers',  required=False, type=int, default=2)
    

    parser.add_argument('--batch_size', required=False, type=int,   default=32,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=5e-5, help='Learning Rate' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=0, help='seed for reproducibility')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument('--top_k',       required=False, type=int,   default=1, help='Top k for generation' )
    parser.add_argument('--model_type', required=False, type=str,   default="lstm", help='Model type' )
    parser.add_argument("--continue_training", required=False, action='store_true',  help='Continue training from the last checkpoint')
    parser.add_argument("--gpt_path", type=str, default="gpt2", help="Path to the GPT model")
    parser.add_argument("--gpt_type", type=str, default="gpt2", help="Type of gpt")

    parser.add_argument("--stage", type=str, choices=["full", "caption", "spotting", "dvc"],
                        default="full", help="Pipeline stage to run. Official order: spotting -> caption")

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))

    if args.wandb:
        if wandb is None:
            raise ModuleNotFoundError("wandb is not installed. Install it or run without --wandb.")
        run = wandb.init(
        project="dvc-res",
        #name=args.model_name,
        settings=wandb.Settings(start_method="fork")
        )

        wandb.config.update(args)

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    run_selected_stages(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
