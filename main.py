import os
import sys
import time
import torch.nn as nn
from datetime import timedelta
import argparse
import torch
import torch.multiprocessing
import os.path as osp
import tabulate
from termcolor import colored
import copy
sys.path.append('/home/server1-ailab/Desktop/Bach/Text_Image_Matching/')
from configs.default import get_default_config
from models import build_model
#from models.gnn import graph_net, utils, trainer, networks, preprocess
from preprocessing.transforms import *
from utils import  Logger
from utils import set_seed
from datasets.build import build_dataloader
from optimizer.build import build_vanilla_optimizer, FreezeBackbone
from tools.evaluate import evaluate
from tools.train import train_epoch, train_epoch_multiview, train_epoch_multiview_gnn, train_epoch_multigrained, train_epoch_multiview_mlm, train_epoch_mlm


import wandb
wandb.login(key="016c2b13617e9b971749b50884ef7bdf9ec68f1a")

best_r1_eval = 0.
best_r1_eval_by_test = 0.

table = []
max_record = ['Max', 'Max', 0, 0, 0, 0, 0]
header = ['Method', 'Dataset', 'Epoch', 'R-1', 'R-5', 'R-10', 'mAP']
table.append(header)
print_interval = 20  # csv
save_interval = 200   # checkpoint


def results_record(name, dataset, epoch, r1, r5, r10, mAP, is_test=False):
    # result csv
    record = list()
    # name = args.name
    record.append(name)
    record.append(dataset)
    record.append(epoch)
    record.append(r1)
    record.append(r5)
    record.append(r10)
    record.append(mAP)

    print_table = copy.deepcopy(table)
    global max_record
    if is_test and record[3] > max_record[3]:
        max_record = copy.deepcopy(record)
        max_record[2] = 'Max_' + str(max_record[2])
    print_table.append(max_record)

    display = tabulate.tabulate(
        print_table,
        tablefmt="pipe",
        headers='firstrow',
        numalign="left",
        floatfmt='.3f')
    print(f"====> results in csv format: \n" + colored(display, "cyan"))


def prepare_start():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                        help='config_file')
    parser.add_argument('--name', default="baseline", type=str, 
                        help='experiments')
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='logs/')
    parser.add_argument('--eval_only', '-eval', action='store_true', help='only eval')
    parser.add_argument('--use_wandb', type=bool, help='use weight and biases', default=False)
    parser.add_argument('--rerank_top', type=int, help='Rerank', default=20)
    parser.add_argument('--use_cuda', type=bool, help='use cuda', default=True)

    parser.add_argument(
        "opts", 
        help="Modify config options using the command-line", 
        default=None, 
        nargs=argparse.REMAINDER, 
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    args.cfg = cfg

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    print(f"====> load config from {args.config}")

    return args, cfg

def main():
    args, cfg = prepare_start()

    set_seed(cfg.TRAIN.SEED, cfg.TRAIN.DETERMINISTIC)
    os.makedirs(args.logs_dir, exist_ok=True)

    if args.use_wandb:
        print("Using Weight and Biases for logging")
        wandb.init(project="Text_Image_Matching",name=cfg.MODEL.NAME, config=cfg)

    # Cross-Modal Model
    trainloader, val_text_loader, val_image_loader, train_data = build_dataloader(cfg)
    if cfg.MODEL.NUM_CLASS == 0:
        cfg.MODEL.NUM_CLASS = len(train_data)

    if args.eval_only:
        args.resume = True
    model = build_model(cfg, args)
    # print(model)

    model_freeze = FreezeBackbone(model, freeze_epoch=cfg.TRAIN.FREEZE_EPOCH)
    optimizer, scheduler = build_vanilla_optimizer(cfg, model, trainloader)

    if args.eval_only:
        evaluate(model, val_text_loader, val_image_loader, 0, args, optimizer)
        return
    
    # Training
    start_time = time.monotonic()
    model.train()
    global_step = 0
    model_freeze.start_freeze_backbone()
    selected_idx = None
    train_fn = None
    train_fn_gnn = None
    
    scaler = torch.cuda.amp.GradScaler()

    if cfg.TRAIN.MODE == 'clip_multiview_gnn':
        print("USING CLIP_MULTIVIEW_GNN TRAINING")
        train_fn_gnn = train_epoch_multiview_gnn(args=args, step=0, cfg=cfg, model=model, optimizer=optimizer, model_freeze=model_freeze, scheduler=scheduler, 
                    trainloader=trainloader, val_image_loader=val_image_loader, val_text_loader=val_text_loader,
                    global_step=global_step, result_record=results_record, save_interval=save_interval, wandb=wandb if args.use_wandb else None, v=selected_idx)
        losses, global_step, metrics = train_fn_gnn.train()
    elif cfg.TRAIN.MODE == 'multiview':
        print("USING MULTIVIEW TRAINING")
        train_fn = train_epoch_multiview
    elif cfg.TRAIN.MODE == 'multiview_mlm':
        print("USING MULTIVIEW MLM TRAINING")
        train_fn = train_epoch_multiview_mlm
    elif cfg.TRAIN.MODE == 'clip_mlm':
        print("USING CLIP MLM")
        train_fn = train_epoch_mlm
    elif cfg.TRAIN.MODE == 'multigrained':
        print("USING MULTIGRAINED TRAINING")
        train_fn = train_epoch_multigrained
    else:
        print("USING DEFAULT TRAINING")
        train_fn = train_epoch
        
    if train_fn is not None:
        for epoch in range(cfg.TRAIN.EPOCH):
            epo_start_time = time.monotonic()
            if cfg.TRAIN.MODE == 'multigrained':
                losses, global_step, metrics = train_fn(cfg=cfg, epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, 
                        trainloader=trainloader, val_image_loader=val_image_loader, val_text_loader=val_text_loader,
                        global_step=global_step, result_record=results_record, args=args, wandb=wandb if args.use_wandb else None)
            else:
                losses, global_step, metrics = train_fn(cfg=cfg, epoch=epoch, model=model, optimizer=optimizer, model_freeze=model_freeze, scheduler=scheduler, 
                        trainloader=trainloader, val_image_loader=val_image_loader, val_text_loader=val_text_loader,
                        global_step=global_step, result_record=results_record, args=args, wandb=wandb if args.use_wandb else None, scaler=scaler)
            
            if epoch % save_interval == 1:
                checkpoint_file = args.logs_dir + "/checkpoint_%d.pth" % epoch
                torch.save({"epoch": epoch,"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_file)

            scheduler.step()
            epo_end_time = time.monotonic()
            print(f'Epoch {epoch} running time: ', timedelta(seconds=epo_end_time - epo_start_time))
            print(f'Logs dir: {args.logs_dir} ')

            if args.use_wandb:
                wandb.log({"AVG Loss": losses})

    del train_data, trainloader
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
    print(f'Logs dir: {args.logs_dir} ')

if __name__ == "__main__":
    main()
    


