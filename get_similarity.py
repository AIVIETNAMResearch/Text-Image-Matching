import torch
from tqdm import tqdm
from datasets import CUHKPEDESDataset, build_transforms
from torch.utils.data import DataLoader
from configs.default import get_default_config
from models.build import build_model
import torch.nn.functional as F
import numpy as np
import argparse

def main(args):
    args.resume = True
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)

    text_data = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                    transform=transform_test, split='train', mode='text')

    text_loader = DataLoader(dataset=text_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)

    model = build_model(cfg, args)
    model.eval()

    all_text_features = []
    all_text_pids = []

    with torch.no_grad():
        for batch_idx, (caption_pids, captions) in tqdm(enumerate(text_loader)):
            text_features = model.encode_text(captions.cuda())
            #text_features = text_features[torch.arange(text_features.shape[0]), captions.argmax(dim=-1)].float()
            all_text_features.append(text_features)
            all_text_pids.append(caption_pids.view(-1))

        all_text_features = torch.cat(all_text_features, dim=0)
        all_text_pids = torch.cat(all_text_pids, dim=0)    

    all_text_features = F.normalize(all_text_features, dim=1, p=2)
    
    similarity = torch.mm(all_text_features, all_text_features.t())
    pid_dist = all_text_pids - all_text_pids.t()
    mask = pid_dist != 0

    similarity = similarity.cpu() * mask.float()

    np.save('./data/CUHK-PEDES/text_sim_matrix.npy', similarity.numpy())
    return similarity


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', default="configs/clip_multiprompt.yaml", type=str,
                        help='config_file')
    parser.add_argument('--use_cuda', type=bool, help='use cuda', default=True)

    main(args=parser.parse_args())
