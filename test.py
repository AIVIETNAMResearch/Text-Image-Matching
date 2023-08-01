import os
import sys
import os.path as osp
import argparse
from configs.default import get_default_config
from utils import  Logger
from utils import set_seed
from datasets.build import build_test_dataloader
from models import build_model
from tools.evaluate import evaluate
import tabulate
import copy
from termcolor import colored

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
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--resume', '-r', type=bool, help='resume from checkpoint', default=True)
    parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                        help='config_file')
    parser.add_argument('--name', default="baseline", type=str, 
                        help='experiments')
    parser.add_argument('--eval_only', '-eval', action='store_true', help='only eval')
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='logs/')
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

def test():
    args, cfg = prepare_start()

    set_seed(cfg.TRAIN.SEED, cfg.TRAIN.DETERMINISTIC)
    os.makedirs(args.logs_dir, exist_ok=True)

    val_text_loader, val_image_loader = build_test_dataloader(cfg)

    if args.eval_only:
        args.resume = True
    model = build_model(cfg, args)

    if args.eval_only:
        evaluate(model, val_text_loader, val_image_loader, 0, args, results_record=results_record)
        

if __name__ == "__main__":
    test()