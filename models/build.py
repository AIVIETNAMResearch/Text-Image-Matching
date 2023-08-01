import torch
from torch.backends import cudnn

from .siamese_baseline import SiameseBaselineModel
from .clip_model import build_CLIP_from_openai_pretrained
from .multi_clip import MultiViewCLIP
from .multiprompt_clip import MultiPromptCLIP
from .multi_clip_gnn import MultiViewCLIPGNN
from .multigrained_clip import MultiGrainedCLIP
from .multi_clip_mlm import MultiViewCLIPMLM, MultiViewCLIPMLMMAE
from .clip_mlm import CLIPMLM

def build_model(cfg, args):
    if cfg.MODEL.NAME == "baseline":
        model = SiameseBaselineModel(cfg.MODEL)
    elif cfg.MODEL.NAME == "clip":
        model, model_cfg = build_CLIP_from_openai_pretrained(cfg.MODEL.PRETRAINED, cfg.DATA.SIZE, cfg.MODEL.STRIDE)
    elif cfg.MODEL.NAME == "clip_mlm":
        model = CLIPMLM(cfg)
    elif cfg.MODEL.NAME == "clip_multiview":
        model = MultiViewCLIP(cfg)
    elif cfg.MODEL.NAME == "clip_multiview_mlm":
        model = MultiViewCLIPMLM(cfg)
    elif cfg.MODEL.NAME == "clip_multiview_mlm_mae":
        model = MultiViewCLIPMLMMAE(cfg)
    elif cfg.MODEL.NAME == "clip_multiview_gnn":
        model = MultiViewCLIPGNN(cfg)
    elif cfg.MODEL.NAME == "clip_multi_prompt":
        model = MultiPromptCLIP(cfg)
    elif cfg.MODEL.NAME == "clip_multigrained":
        model = MultiGrainedCLIP(cfg)
    else:
        assert False, "Unknown model name: {}".format(cfg.MODEL.NAME)
    
    if args.resume:
        assert cfg.TEST.RESTORE_FROM != None, "Please specify the checkpoint to resume from."
        print("RESTORE FROM ",cfg.TEST.RESTORE_FROM)
        checkpoint = torch.load(cfg.TEST.RESTORE_FROM, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Loaded checkpoint from {}".format(cfg.TEST.RESTORE_FROM))
    else:
        print("Training from scratch.")

    if args.use_cuda:
        model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #cudnn.benchmark = cfg.TRAIN.BENCHMARK

    return model