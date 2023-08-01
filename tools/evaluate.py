import torch
import time
from tqdm import tqdm
from utils import AverageMeter, ProgressMeter
import torch.nn.functional as F
from utils.metrics import rank
from datetime import timedelta
import gc
import numpy as np
best_r1_eval = 0.
best_r1_eval_by_test = 0.

def evaluate(model, val_text_loader, val_img_loader, epoch, args=None, optimizer=None, results_record=None, cross_encoder=False, test_bs=513):
    """ evaluate merge features"""
    global best_r1_eval_by_test
    print(f"====> Test::::{val_text_loader.dataset.name}")
    evl_start_time = time.monotonic()
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1_rank = AverageMeter('Rank@1', ':6.4f')
    top5_rank = AverageMeter('Rank@5', ':6.4f')
    top10_rank = AverageMeter('Rank@10', ':6.4f')
    mAP = AverageMeter('mAP', ':6.4f')
    if cross_encoder:
        top1_rerank = AverageMeter('Rank@1_rereank', ':6.4f')

    progress = ProgressMeter(
        len(val_text_loader),
        [batch_time, data_time, top1_rank, top5_rank, top10_rank, mAP, top1_rank if cross_encoder else None],
        prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()

    all_visual_feature = []
    all_visual_pids = []
    all_text_features = []
    all_text_pids = []

    all_ori_text_features = []
    all_ori_image_features = []
    all_text_caps = []

    with torch.no_grad():
        for batch_idx, (caption_pids, captions) in tqdm(enumerate(val_text_loader)):
            ori_text_features = model.encode_text(captions.cuda(), return_dense=True)
            text_features = ori_text_features[torch.arange(ori_text_features.shape[0]), captions.argmax(dim=-1)].float()
            all_text_caps.append(captions)

            all_text_features.append(text_features)
            all_text_pids.append(caption_pids.view(-1))
            all_ori_text_features.append(ori_text_features)

        for batch_idx, (img_pids, imgs) in tqdm(enumerate(val_img_loader)):
            ori_visual_features = model.encode_image(imgs.cuda(), return_dense=True)
            visual_features = ori_visual_features[:, 0, :].float()
            
            all_visual_feature.append(visual_features)
            all_visual_pids.append(img_pids.view(-1))
            all_ori_image_features.append(ori_visual_features)

        all_visual_feature = torch.cat(all_visual_feature, dim=0)
        all_visual_pids = torch.cat(all_visual_pids, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        all_text_pids = torch.cat(all_text_pids, dim=0)  
        all_ori_text_features = torch.cat(all_ori_text_features)
        all_ori_image_features = torch.cat(all_ori_image_features)  
        all_text_caps = torch.cat(all_text_caps, dim=0)

    all_visual_feature = F.normalize(all_visual_feature, dim=1, p=2)
    all_text_features = F.normalize(all_text_features, dim=1, p=2)    
    similarity = torch.mm(all_text_features, all_visual_feature.t())
    rerank_similarity = torch.zeros_like(similarity)

    del all_visual_feature, all_text_features
    gc.collect()

    batch_size = test_bs
    if cross_encoder:
        with torch.no_grad():
            for i in tqdm(range(0, len(all_ori_text_features), batch_size), total=len(all_ori_text_features) // batch_size):
                for j in range(0, len(all_ori_image_features), batch_size):
                    text_features = all_ori_text_features[i:i+batch_size]
                    image_features = all_ori_image_features[j:j+batch_size]
                    text_caps = all_text_caps[i:i+batch_size]
                    rerank_similarity[i:i+batch_size, j:j+batch_size] = model.rerank(text_features, image_features, text_caps).cpu()

    t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity, q_pids=all_text_pids, g_pids=all_visual_pids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    rank1, rank5, rank10 = t2i_cmc[0], t2i_cmc[4], t2i_cmc[9]
    
    t2i_cmc_rr, t2i_mAP_rr, t2i_mINP_rr, _ = rank(rerank_similarity, q_pids=all_text_pids, g_pids=all_visual_pids, max_rank=10, get_mAP=True)
    t2i_cmc_rr, t2i_mAP_rr, t2i_mINP_rr = t2i_cmc_rr.numpy(), t2i_mAP_rr.numpy(), t2i_mINP_rr.numpy()

    if cross_encoder:
        rank_1_rr, rank5_rr, rank_10_rr = t2i_cmc_rr[0], t2i_cmc_rr[4], t2i_cmc_rr[9]
    
    top1_rank.update(rank1)
    top5_rank.update(rank5)
    top10_rank.update(rank10)
    
    if cross_encoder:
        top1_rerank.update(rank_1_rr)

    batch_time.update(time.time() - end)
    progress.display(batch_idx)

    evl_end_time = time.monotonic()
    print(f'Epoch {epoch} running time: ', timedelta(seconds=evl_end_time - evl_start_time))
    print(f'Logs dir: {args.logs_dir} ')

    results_record(args.logs_dir.split('/')[-1], val_text_loader.dataset.name, epoch, rank1, rank5, rank10, t2i_mAP, is_test=True)
    if args.eval_only:
        return
    if rank1 > best_r1_eval_by_test or (cross_encoder and rank_1_rr > best_r1_eval_by_test):
        # save time
        best_r1_eval_by_test = rank1
        checkpoint_file = args.logs_dir + "/checkpoint_best_eval.pth"
        torch.save({"epoch": epoch,"state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None}, checkpoint_file)
        print(f"====> save checkpoint to {checkpoint_file}")

    if cross_encoder:
        return rank1, rank5, rank10, t2i_mAP, rank_1_rr
    return rank1, rank5, rank10, t2i_mAP

