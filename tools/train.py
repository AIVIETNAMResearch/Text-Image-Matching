from utils import AverageMeter, ProgressMeter
from loss import compute_sdm, compute_clip, SimsiamLoss, FocalLoss, compute_id, compute_triplet, compute_mlm
import torch
from torch import nn
from .evaluate import evaluate
import time
from datetime import timedelta
from torch.nn import functional as F
from models.gnn import label2edge, transform_shape, Discriminator
from optimizer.build import build_vanilla_optimizer_combine
from torch.autograd import Variable
from models import gnn
from loss.focal import FocalLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(cfg, epoch, model, model_freeze, trainloader, 
                val_text_loader, val_image_loader, optimizer, args, 
                global_step, scheduler,result_record, wandb=None):

    rank1, rank5, rank10, mAP = None, None, None, None
    if cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (cfg.EVAL.EPOCH * cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
        rank1, rank5, rank10, mAP = evaluate(model, val_text_loader, val_image_loader, epoch, args, optimizer, result_record)
        if wandb is not None:
            wandb.log({"Rank-1": rank1, "Rank-5": rank5, "Rank-10": rank10, "mAP": mAP, "Epoch": epoch})

    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT, 
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        model_freeze.on_train_epoch_start(epoch=epoch * cfg.TRAIN.ONE_EPOCH_REPEAT + tmp)
        for batch_idx, batch in enumerate(trainloader):

            image, caption, pid = batch["image"], batch["caption"], batch["pid"]
            data_time.update(time.time() - end)
            global_step += 1
            visual_features, text_features, logit_scale = model(image.cuda(), caption.cuda())

            clip_loss = torch.tensor(0., device='cuda')
            sdm_loss = torch.tensor(0., device='cuda')
            
            if cfg.MODEL.HEAD.CLIP_LOSS:
                sim_i_2_t = torch.matmul(visual_features, torch.t(text_features))
                acc_sim = sim_i_2_t.clone().detach()
                sim_i_2_t = sim_i_2_t - (torch.eye(image.size(0)).cuda() * cfg.MODEL.HEAD.CLIP_LOSS_MARGIN)
                sim_i_2_t = torch.mul(logit_scale, sim_i_2_t)
                sim_t_2_i = sim_i_2_t.t()
                loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
                loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())
                clip_loss = (loss_t_2_i+loss_i_2_t)/2
            
            if cfg.MODEL.HEAD.SDM_LOSS:
                sdm_loss = compute_sdm(visual_features, text_features, pid.cuda(), logit_scale)
            
            loss = clip_loss + sdm_loss
            optimizer.zero_grad()

            losses.update(loss.item(), image.size(0))
            loss.backward()
            optimizer.step()

            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step % (len(trainloader) * 30))

            if wandb is not None:
                wandb.log({'Loss': loss})

    return losses.avg, global_step, (rank1, rank5, rank10, mAP)


def train_epoch_multiview(cfg, epoch, model, model_freeze, trainloader, 
                val_text_loader, val_image_loader, optimizer, args, 
                global_step, scheduler,result_record, wandb=None):

    simsiam_criterion = SimsiamLoss()
    rank1, rank5, rank10, mAP = None, None, None, None
    if cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (cfg.EVAL.EPOCH * cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
        rank1, rank5, rank10, mAP = evaluate(model, val_text_loader, val_image_loader, epoch, args, optimizer, result_record)
        if wandb is not None:
            wandb.log({"Rank-1": rank1, "Rank-5": rank5, "Rank-10": rank10, "mAP": mAP, "Epoch": epoch})

    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT, 
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        model_freeze.on_train_epoch_start(epoch=epoch * cfg.TRAIN.ONE_EPOCH_REPEAT + tmp)
        for batch_idx, batch in enumerate(trainloader):

            (image1, image2), (caption1, caption2), pid = batch["images"], batch["captions"], batch["pid"]
            data_time.update(time.time() - end)
            global_step += 1
            output = model(image1.cuda(), image2.cuda(), caption1.cuda(), caption2.cuda())
            (image_feats1, image_feats2) = output['image_feats']
            (text_feats1, text_feats2) = output['text_feats']
            (z1_image, z2_image, p1_image, p2_image) = output['simsiam_features_images']
            (z1_text, z2_text, p1_text, p2_text) = output['simsiam_features_texts']
            logit_scale = output['logit_scale']
            (image_logits_1, text_logits_1, image_logits_2, text_logits_2) = output['id_logits']
            #(image_logits_1, text_logits_1) = output['id_logits']
            #(cross_image_features, cross_text_features) = output['cross_features']
            
            image_feats1 = F.normalize(image_feats1, dim=1)
            image_feats2 = F.normalize(image_feats2, dim=1)
            text_feats1 = F.normalize(text_feats1, dim=1)
            text_feats2 = F.normalize(text_feats2, dim=1)

            clip_loss = torch.tensor(0., device='cuda')
            sdm_loss = torch.tensor(0., device='cuda')
            sdm_cross = torch.tensor(0., device='cuda')
            simsiam_loss = torch.tensor(0., device='cuda')
            id_loss = torch.tensor(0., device='cuda')
            triplet_loss = torch.tensor(0., device='cuda')


            if cfg.MODEL.HEAD.CLIP_LOSS:
                clip_loss_11 = compute_clip(image_feats1, text_feats1, pid.cuda(), logit_scale, image1.size(0))
                clip_loss_12 = compute_clip(image_feats1, text_feats2, pid.cuda(), logit_scale, image1.size(0))
                clip_loss_21 = compute_clip(image_feats2, text_feats1, pid.cuda(), logit_scale, image1.size(0))
                clip_loss_22 = compute_clip(image_feats2, text_feats2, pid.cuda(), logit_scale, image1.size(0))
                clip_loss = 0.6 * (clip_loss_11 + clip_loss_12) + 0.4 * (clip_loss_21 + clip_loss_22)

            if cfg.MODEL.HEAD.SDM_LOSS:
                sdm_loss_11 = compute_sdm(image_feats1, text_feats1, pid.cuda(), logit_scale)  
                sdm_loss_12 = compute_sdm(image_feats1, text_feats2, pid.cuda(), logit_scale)
                sdm_loss_21 = compute_sdm(image_feats2, text_feats1, pid.cuda(), logit_scale)
                sdm_loss_22 = compute_sdm(image_feats2, text_feats2, pid.cuda(), logit_scale)
                sdm_loss = (sdm_loss_11 + sdm_loss_12 + sdm_loss_21 + sdm_loss_22) / 4

                # SDM for cross feaetures
                #sdm_cross = compute_sdm(cross_image_features, cross_text_features, pid.cuda(), logit_scale)

            if cfg.MODEL.HEAD.SIMSIAM_LOSS:
                text_simsiam_loss = simsiam_criterion(p1_text, z1_text, p2_text, z2_text)
                image_simsiam_loss = simsiam_criterion(p1_image, z1_image, p2_image, z2_image) 
                simsiam_loss = (text_simsiam_loss + image_simsiam_loss) 

            if cfg.MODEL.HEAD.ID_LOSS:
                id_loss_1 = compute_id(image_logits_1, text_logits_1, pid.cuda())
                id_loss_2 = compute_id(image_logits_2, text_logits_2, pid.cuda())
                #id_loss_cross = compute_id(cross_img_logits, cross_txt_logits, pid.cuda())
                id_loss = (id_loss_1 + id_loss_2) / 2

            if cfg.MODEL.HEAD.TRIPLET_LOSS:
                triplet_loss_11 = compute_triplet(image_feats1, text_feats1, pid.cuda())
                triplet_loss_12 = compute_triplet(image_feats1, text_feats2, pid.cuda())
                triplet_loss_21 = compute_triplet(image_feats2, text_feats1, pid.cuda())
                triplet_loss_22 = compute_triplet(image_feats2, text_feats2, pid.cuda())

                triplet_loss = (triplet_loss_11 + triplet_loss_12 + triplet_loss_21 + triplet_loss_22) / 4

            loss = clip_loss + sdm_loss + simsiam_loss + id_loss + triplet_loss + sdm_cross
            optimizer.zero_grad()

            losses.update(loss.item(), image1.size(0))
            loss.backward()
            optimizer.step()

            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step % (len(trainloader) * 30))

            if wandb is not None:
                wandb.log({'Loss': loss, 'SDM Loss': sdm_loss, 'SimSiamLoss': simsiam_loss, 
                           'CLIP Loss': clip_loss, 'ID Loss': id_loss, 'Triplet Loss': triplet_loss,
                           'lr': optimizer.param_groups[0]['lr']})
    return losses.avg, global_step, (rank1, rank5, rank10, mAP)


class train_epoch_multiview_gnn():
    def __init__(self, args, step, cfg, model, optimizer, model_freeze, scheduler, 
                    trainloader, val_image_loader, val_text_loader,
                    global_step, result_record, save_interval, wandb, v = None):
        self.args = args
        self.step = step
        # self.data = data
        # self.label_flag = label_flag

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.model_freeze = model_freeze
        self.scheduler =scheduler
        self.trainloader = trainloader
        self.val_image_loader = val_image_loader
        self.val_text_loader = val_text_loader
        self.global_step = global_step
        self.result_record = result_record
        self.save_interval = save_interval
        self.wandb = wandb

        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.epochs = self.cfg.TRAIN.EPOCH
        self.in_feature_shape = 512
        self.data_workers = 6

        # self.num_class = data.num_class
        # self.num_task = args.batch_size
        # self.num_to_select = 0

        # self.model = gnn.create("gnn", self.args)
        # self.model = nn.DataParallel(self.model).cuda()

        #GNN
        self.gnnModel = gnn.create('gnn', self.args, self.cfg)
        self.gnnModel = nn.DataParallel(self.gnnModel).cuda()

        # self.meter = meter(args.num_class)
        # self.v = v

        # CE for node classification
        self.criterionCE = FocalLoss().cuda()
        # if args.loss == 'focal':
        #     self.criterionCE = FocalLoss().cuda()
        # elif args.loss == 'nll':
        #     self.criterionCE = nn.NLLLoss(reduction='mean').cuda()

        # BCE for edge
        self.criterion = nn.BCELoss(reduction='mean').cuda()
        # self.global_step = 0
        # self.logger = logger
        self.val_acc = 0
        # self.threshold = self.args.threshold

        if self.cfg.TRAIN.DISCRIMINATOR:
            self.discriminator = Discriminator(self.in_feature_shape)
            self.discriminator = nn.DataParallel(self.discriminator).cuda()

    def adjust_lr(self, epoch, step_size):
        lr = self.cfg.TRAIN.LR.BASE_LR / (2 ** (epoch // step_size))
        for g in self.optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

        if epoch % step_size == 0:
            print("Epoch {}, current lr {}".format(epoch, lr))

    def train(self, step=1):
        param_groups = [
                        {'params': self.model.parameters(), 'lr_mult': 0.01},
                        {'params': self.gnnModel.parameters(), 'lr_mult': 0.1},
                    ]
        if self.cfg.TRAIN.DISCRIMINATOR:
            param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 0.1})

        combine_optimizer, combine_scheduler = build_vanilla_optimizer_combine(cfg=self.cfg, params_group=param_groups, trainloader=self.trainloader)

        self.gnnModel.train()
        # self.meter.reset()
        for epoch in range(self.epochs):
            self.adjust_lr(epoch, step)
            simsiam_criterion = SimsiamLoss()
            rank1, rank5, rank10, mAP = None, None, None, None
            if self.cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (self.cfg.EVAL.EPOCH * self.cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
                rank1, rank5, rank10, mAP = evaluate(self.model, self.val_text_loader, self.val_image_loader, epoch, self.args, self.optimizer, self.result_record)

            self.model.train()
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')

            progress = ProgressMeter(
                len(self.trainloader)*self.cfg.TRAIN.ONE_EPOCH_REPEAT, 
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))
            end = time.time()
        
            epo_start_time = time.monotonic()
            for tmp in range(self.cfg.TRAIN.ONE_EPOCH_REPEAT):
                self.model_freeze.on_train_epoch_start(epoch=epoch * self.cfg.TRAIN.ONE_EPOCH_REPEAT + tmp)
                for batch_idx, batch in enumerate(self.trainloader):

                    (image1, image2), (caption1, caption2), pid = batch["images"], batch["captions"], batch["pid"]
                    data_time.update(time.time() - end)
                    self.global_step += 1
                    output = self.model(image1.cuda(), image2.cuda(), caption1.cuda(), caption2.cuda())
                    (image_feats1, image_feats2) = output['image_feats']
                    (text_feats1, text_feats2) = output['text_feats']
                    (z1_image, z2_image, p1_image, p2_image) = output['simsiam_features_images']
                    (z1_text, z2_text, p1_text, p2_text) = output['simsiam_features_texts']
                    logit_scale = output['logit_scale']

                    # if self.args.discriminator:
                    #     domain_label = Variable(inputs[3].float()).cuda()
                    print("batch['pid']: ", batch["pid"], batch["pid"].shape)
                    # feed into gnn networks
                    targets_batch = Variable(batch["pid"]).cuda()
                    print("targets_batch: ", targets_batch, targets_batch.shape)

                    targets = transform_shape(targets_batch.unsqueeze(-1)).squeeze(-1)
                    p1_image_feature = transform_shape(p1_image)
                    p2_image_feature = transform_shape(p2_image)
                    init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = label2edge(targets=targets_batch, tensor_shape=512)
                    edge_logits, node_logits = self.gnnModel(init_node_feat=p1_image_feature, init_edge_feat=init_edge,
                                                                    target_mask=target_edge_mask)

                    # init contrastive loss
                    clip_loss = torch.tensor(0., device='cuda')
                    sdm_loss = torch.tensor(0., device='cuda')
                    simsiam_loss = torch.tensor(0., device='cuda')
                    criterionCE = FocalLoss().cuda()
                    
                    # init GNN loss
                    criterionCE = FocalLoss().cuda()
                    criterion = nn.BCELoss(reduction='mean').cuda()
                    discriminator = Discriminator(inc=512)
                    discriminator = nn.DataParallel(discriminator).cuda()

                    # Contrastive loss
                    if self.cfg.MODEL.HEAD.CLIP_LOSS:
                        clip_loss_11 = compute_clip(image_feats1, text_feats1, pid.cuda(), logit_scale, image1.size(0))
                        clip_loss_12 = compute_clip(image_feats1, text_feats2, pid.cuda(), logit_scale, image1.size(0))
                        clip_loss_21 = compute_clip(image_feats2, text_feats1, pid.cuda(), logit_scale, image1.size(0))
                        clip_loss_22 = compute_clip(image_feats2, text_feats2, pid.cuda(), logit_scale, image1.size(0))
                        clip_loss = 0.6 * (clip_loss_11 + clip_loss_12) + 0.4 * (clip_loss_21 + clip_loss_22)
                    if self.cfg.MODEL.HEAD.SDM_LOSS:
                        sdm_loss_11 = compute_sdm(image_feats1, text_feats1, pid.cuda(), logit_scale)  
                        sdm_loss_12 = compute_sdm(image_feats1, text_feats2, pid.cuda(), logit_scale)
                        sdm_loss_21 = compute_sdm(image_feats2, text_feats1, pid.cuda(), logit_scale)
                        sdm_loss_22 = compute_sdm(image_feats2, text_feats2, pid.cuda(), logit_scale)
                        sdm_loss = 0.6 * (sdm_loss_11 + sdm_loss_12) + 0.4 * (sdm_loss_21 + sdm_loss_22)
                    if self.cfg.MODEL.HEAD.SIMSIAM_LOSS:
                        text_simsiam_loss = simsiam_criterion(p1_text, z1_text, p2_text, z2_text)
                        image_simsiam_loss = simsiam_criterion(p1_image, z1_image, p2_image, z2_image) 
                        simsiam_loss = (text_simsiam_loss + image_simsiam_loss) / 2
                        
                    # GNN loss
                    # compute edge_loss
                    full_edge_loss = [criterion(edge_logit.masked_select(source_edge_mask), init_edge.masked_select(source_edge_mask)) for edge_logit in edge_logits]
                    norm_node_logits = F.softmax(node_logits[-1], dim=-1)
                    source_node_loss = criterionCE(norm_node_logits[source_node_mask, :], targets.masked_select(source_node_mask))
                    edge_loss = 0
                    for l in range(self.cfg.TRAIN.BATCH_SIZE - 1):
                        edge_loss += full_edge_loss[l] * 0.5
                    edge_loss += full_edge_loss[-1] * 1
                    # compute node_loss
                    node_loss = 0.3
                    # total gnn_loss
                    gnn_loss = 1 * edge_loss + node_loss* source_node_loss

                    # if self.cfg.TRAIN.DISCRIMINATOR:
                    #     unk_label_mask = torch.eq(targets, self.cfg.TRAIN.BATCH_SIZE).squeeze()
                    #     domain_pred = self.discriminator(p1_image_feature)
                    #     temp = domain_pred.view(-1)[~unk_label_mask]
                    #     domain_loss = self.criterion(temp, domain_label.view(-1)[~unk_label_mask]) #(targets.size(1) / temp.size(0)) *
                    #     loss = loss + self.cfg.TRAIN.ADV_COEFF * domain_loss

                    node_pred = norm_node_logits[source_node_mask, :].detach().cpu().max(1)[1]
                    node_prec = node_pred.eq(targets.masked_select(source_node_mask).detach().cpu()).double().mean()

                    # Total loss
                    loss = clip_loss + sdm_loss + simsiam_loss + gnn_loss
                    self.optimizer.zero_grad()

                    losses.update(loss.item(), image1.size(0))
                    loss.backward()
                    # self.optimizer.step()
                    combine_optimizer.step()

                    # self.scheduler.step()
                    combine_scheduler.step()
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if batch_idx % self.cfg.TRAIN.PRINT_FREQ == 0:
                        progress.display(global_step % (len(self.trainloader) * 30))

            if epoch % self.save_interval == 1:
                checkpoint_file = self.args.logs_dir + "/checkpoint_%d.pth" % epoch

            epo_end_time = time.monotonic()
            print(f'Epoch {epoch} running time: ', timedelta(seconds=epo_end_time - epo_start_time))
            print(f'Logs dir: {self.args.logs_dir} ')

            if self.args.use_wandb:
                self.wandb.log({'Loss': loss, "Rank-1": rank1, "Rank-5": rank5, "Rank-10": rank10, "mAP": mAP, "AVG Loss": losses, "Epoch": epoch})
            torch.save({"epoch": epoch,"state_dict": self.model.state_dict(), "gnn_state_dict": self.gnnModel, "optimizer": self.combine_optimizer.state_dict()}, checkpoint_file)

        return losses.avg, global_step, (rank1, rank5, rank10, mAP)


def train_epoch_multigrained(cfg, epoch, model, trainloader, 
                val_text_loader, val_image_loader, optimizer, args, 
                global_step, scheduler,result_record, wandb=None):

    rank1, rank5, rank10, mAP = None, None, None, None
    if cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (cfg.EVAL.EPOCH * cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
        rank1, rank5, rank10, mAP= evaluate(model, val_text_loader, val_image_loader, epoch, args, 
                                             optimizer, result_record)
        if wandb is not None:
            wandb.log({"Rank-1": rank1, "Rank-5": rank5, "Rank-10": rank10, "mAP": mAP, "Epoch": epoch})

    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT, 
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        for batch_idx, batch in enumerate(trainloader):
            image, caption, pid = batch["image"], batch["caption"], batch["pid"]
            data_time.update(time.time() - end)
            global_step += 1
            # image_feats = base_model.encode_image(image.cuda())
            # text_feats = base_model.encode_text(caption.cuda())
            res = model(image.cuda(), caption.cuda())

            (image_features, text_features) = res['global_features']
            (cross_image_features, cross_text_features) = res['cross_features']
            logit_scale = res['logit_scale']
            (cross_img_logits, cross_txt_logits) = res['cross_logits']
            (image_logits, text_logits) = res['global_logits']

            optimizer.zero_grad()
            sdm_loss = torch.tensor(0., device='cuda')
            triplet_loss = torch.tensor(0., device='cuda')
            id_loss = torch.tensor(0., device='cuda')

            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
            cross_text_features = F.normalize(cross_text_features, dim=1)
            cross_image_features = F.normalize(cross_image_features, dim=1)

            if cfg.MODEL.HEAD.SDM_LOSS:
                global_sdm_loss = compute_sdm(image_features, text_features, pid.cuda(), logit_scale)  
                cross_sdm_loss = compute_sdm(cross_image_features, cross_text_features, pid.cuda(), logit_scale)
                cross_global_loss = compute_sdm(cross_image_features, text_features, pid.cuda(), logit_scale)
                global_cross_loss = compute_sdm(image_features, cross_text_features, pid.cuda(), logit_scale)
                sdm_loss=  (global_sdm_loss + cross_sdm_loss + cross_global_loss + global_cross_loss) / 4
            if cfg.MODEL.HEAD.ID_LOSS:
                cross_id_loss = compute_id(cross_img_logits, cross_txt_logits, pid.cuda())
                global_id_loss = compute_id(image_logits, text_logits, pid.cuda())
                id_loss = (cross_id_loss + global_id_loss) / 2
            
            if cfg.MODEL.HEAD.TRIPLET_LOSS:
                cross_cross_loss = compute_triplet(cross_image_features, cross_text_features, pid.cuda())
                cross_global_text_loss = compute_triplet(cross_image_features, text_features, pid.cuda())
                cross_global_image_loss = compute_triplet(image_features, cross_text_features, pid.cuda())
                global_global_loss = compute_triplet(image_features, text_features, pid.cuda())
                
                triplet_loss = (cross_cross_loss + cross_global_text_loss + cross_global_image_loss + global_global_loss) / 8
                #triplet_loss = global_global_loss
            loss = sdm_loss + id_loss + triplet_loss
            
            losses.update(loss.item(), image.size(0))
            loss.backward()
            optimizer.step()

            
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step % (len(trainloader) * 30))

            if wandb is not None:
                wandb.log({'Loss': loss, 'SDM Loss': sdm_loss, 'ID Loss': id_loss, "Epoch": epoch, 'Triplet Loss': triplet_loss, "lr": optimizer.param_groups[0]['lr']})

    return losses.avg, global_step, (rank1, rank5, rank10, mAP)



def train_epoch_multiview_mlm(cfg, epoch, model, model_freeze, trainloader, 
                val_text_loader, val_image_loader, optimizer, args, 
                global_step, scheduler,result_record, wandb=None, scaler=None):

    simsiam_criterion = SimsiamLoss()
    rank1, rank5, rank10, mAP = None, None, None, None
    if cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (cfg.EVAL.EPOCH * cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
        rank1, rank5, rank10, mAP = evaluate(model, val_text_loader, val_image_loader, epoch, args, optimizer, result_record)
        if wandb is not None:
            wandb.log({"Rank-1": rank1, "Rank-5": rank5, "Rank-10": rank10, "mAP": mAP, "Epoch": epoch})

    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mlm_accuracy = AverageMeter('MLM Acc', ':6.3f')

    progress = ProgressMeter(
        len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT, 
        [batch_time, data_time, losses, mlm_accuracy],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        model_freeze.on_train_epoch_start(epoch=epoch * cfg.TRAIN.ONE_EPOCH_REPEAT + tmp)
        for batch_idx, batch in enumerate(trainloader):

            (image1, image2), (caption1, caption2), pid = batch["images"], batch["captions"], batch["pid"]
            (mlm_tokens, mlm_labels) = batch["mlm"]
            data_time.update(time.time() - end)
            global_step += 1
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(image1.cuda(), image2.cuda(), caption1.cuda(), caption2.cuda(), mlm_tokens.cuda())
                (image_feats1, image_feats2) = output['image_feats']
                (text_feats1, text_feats2) = output['text_feats']
                (z1_image, z2_image, p1_image, p2_image) = output['simsiam_features_images']
                (z1_text, z2_text, p1_text, p2_text) = output['simsiam_features_texts']
                logit_scale = output['logit_scale']
                (image_logits_1, text_logits_1, image_logits_2, text_logits_2) = output['id_logits']
                mlm_scores = output['mlm_scores']
                
                image_feats1 = F.normalize(image_feats1, dim=1)
                image_feats2 = F.normalize(image_feats2, dim=1)
                text_feats1 = F.normalize(text_feats1, dim=1)
                text_feats2 = F.normalize(text_feats2, dim=1)


                clip_loss = torch.tensor(0., device='cuda')
                sdm_loss = torch.tensor(0., device='cuda')
                mlm_loss = torch.tensor(0., device='cuda')
                simsiam_loss = torch.tensor(0., device='cuda')
                id_loss = torch.tensor(0., device='cuda')
                triplet_loss = torch.tensor(0., device='cuda')
                mae_loss = torch.tensor(0., device='cuda')

                if cfg.MODEL.HEAD.CLIP_LOSS:
                    clip_loss_11 = compute_clip(image_feats1, text_feats1, pid.cuda(), logit_scale, image1.size(0))
                    clip_loss_12 = compute_clip(image_feats1, text_feats2, pid.cuda(), logit_scale, image1.size(0))
                    clip_loss_21 = compute_clip(image_feats2, text_feats1, pid.cuda(), logit_scale, image1.size(0))
                    clip_loss_22 = compute_clip(image_feats2, text_feats2, pid.cuda(), logit_scale, image1.size(0))
                    clip_loss = (clip_loss_11 + clip_loss_12 + clip_loss_21 + clip_loss_22) / 4

                if cfg.MODEL.HEAD.SDM_LOSS:
                    sdm_loss_11 = compute_sdm(image_feats1, text_feats1, pid.cuda(), logit_scale)  
                    sdm_loss_12 = compute_sdm(image_feats1, text_feats2, pid.cuda(), logit_scale)
                    sdm_loss_21 = compute_sdm(image_feats2, text_feats1, pid.cuda(), logit_scale)
                    sdm_loss_22 = compute_sdm(image_feats2, text_feats2, pid.cuda(), logit_scale)
                    if cfg.MODEL.HEAD.SELF_SDM:
                        sdm_loss_im = compute_sdm(image_feats1, image_feats2, pid.cuda(), logit_scale)
                        sdm_loss_tx = compute_sdm(text_feats1, text_feats2, pid.cuda(), logit_scale)
                        sdm_loss = (4 * sdm_loss_11 + 1.5 * sdm_loss_12 + 1.5 * sdm_loss_21 + 1.5 * sdm_loss_22 + sdm_loss_im + sdm_loss_tx) / 10.5
                    else:
                        sdm_loss = (0.7 * (sdm_loss_11 + sdm_loss_12) + 0.3 * (sdm_loss_21 + sdm_loss_22)) / 3

                if cfg.MODEL.HEAD.SIMSIAM_LOSS:
                    text_simsiam_loss = simsiam_criterion(p1_text, z1_text, p2_text, z2_text)
                    image_simsiam_loss = simsiam_criterion(p1_image, z1_image, p2_image, z2_image) 
                    simsiam_loss = (text_simsiam_loss + image_simsiam_loss) 

                if cfg.MODEL.HEAD.ID_LOSS:
                    id_loss_1 = compute_id(image_logits_1, text_logits_1, pid.cuda())
                    id_loss_2 = compute_id(image_logits_2, text_logits_2, pid.cuda())
                    id_loss = 0.6 * id_loss_1 + 0.4 * id_loss_2
                    #id_loss = id_loss_1
                if cfg.MODEL.HEAD.TRIPLET_LOSS:
                    triplet_11 = compute_triplet(image_feats1, text_feats1, pid.cuda())
                    triplet_12 = compute_triplet(image_feats1, text_feats2, pid.cuda())
                    triplet_21 = compute_triplet(image_feats2, text_feats1, pid.cuda())
                    triplet_22 = compute_triplet(image_feats2, text_feats2, pid.cuda())
                    
                    triplet_loss = (2 * triplet_11 + triplet_12 + triplet_21 + triplet_22) / 8
                    
                if cfg.MODEL.HEAD.ENABLE_MLM:
                    mlm_scores = mlm_scores.float().reshape(-1, cfg.MODEL.VOCAB_SIZE)
                    mlm_labels = mlm_labels.reshape(-1).cuda()
                    mlm_loss = compute_mlm(mlm_scores, mlm_labels)
                    
                    pred = mlm_scores.max(1)[1]
                    mlm_label_idx = torch.nonzero(mlm_labels)
                    mlm_acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
                    mlm_accuracy.update(mlm_acc.item(), image1.size(0))

                if cfg.MODEL.HEAD.ENABLE_MAE:
                    target, mae_pred, mask = output['mae']
                    mae_loss = (mae_pred - target) ** 2
                    mae_loss = mae_loss.mean(dim=-1)
                    mae_loss = (mae_loss*mask).sum() / mask.sum()

                loss = clip_loss + sdm_loss + simsiam_loss + id_loss + mlm_loss + triplet_loss + mae_loss
                
                if cfg.TRAIN.ENABLE_GRADIENT_ACCUMULATION:
                    loss = loss / cfg.TRAIN.GRADIENT_ACCUMULATION_STEP

                losses.update(loss.item(), image1.size(0))

            scaler.scale(loss).backward()
            
            accum_iter  = 1
            if cfg.TRAIN.ENABLE_GRADIENT_ACCUMULATION:
                accum_iter  = cfg.TRAIN.GRADIENT_ACCUMULATION_STEP
            
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(trainloader)):              
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.MAX_NORM)
                scaler.step(optimizer)
                scaler.update()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step % (len(trainloader) * 30))

            if wandb is not None:
                wandb.log({'Loss': loss, 'SDM Loss': sdm_loss, 'SimSiamLoss': simsiam_loss, 
                           'ID Loss': id_loss, "mlm_loss": mlm_loss, "mlm_acc": mlm_acc, 
                           'Triplet Loss': triplet_loss, "MAE Loss": mae_loss, 
                           'lr': optimizer.param_groups[0]['lr']})
    return losses.avg, global_step, (rank1, rank5, rank10, mAP)


def train_epoch_mlm(cfg, epoch, model, model_freeze, trainloader, 
                val_text_loader, val_image_loader, optimizer, args, 
                global_step, scheduler,result_record, wandb=None):

    simsiam_criterion = SimsiamLoss()
    rank1, rank5, rank10, mAP = None, None, None, None
    if cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (cfg.EVAL.EPOCH * cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
        rank1, rank5, rank10, mAP = evaluate(model, val_text_loader, val_image_loader, epoch, args, optimizer, result_record)
        if wandb is not None:
            wandb.log({"Rank-1": rank1, "Rank-5": rank5, "Rank-10": rank10, "mAP": mAP, "Epoch": epoch})

    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mlm_accuracy = AverageMeter('MLM Acc', ':6.3f')

    progress = ProgressMeter(
        len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT, 
        [batch_time, data_time, losses, mlm_accuracy],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        model_freeze.on_train_epoch_start(epoch=epoch * cfg.TRAIN.ONE_EPOCH_REPEAT + tmp)
        for batch_idx, batch in enumerate(trainloader):

            image, caption, pid = batch["image"], batch["caption"], batch["pid"]
            (mlm_tokens, mlm_labels) = batch["mlm"]
            
            data_time.update(time.time() - end)
            global_step += 1
            output = model(image.cuda(), caption.cuda(), mlm_tokens.cuda())
            image_feats = output['image_feats']
            text_feats = output['text_feats']
            logit_scale = output['logit_scale']
            image_logits, text_logits = output['id_logits']
            mlm_scores = output['mlm_scores']
            
            image_feats = F.normalize(image_feats, dim=1)
            text_feats = F.normalize(text_feats, dim=1)


            sdm_loss = torch.tensor(0., device='cuda')
            mlm_loss = torch.tensor(0., device='cuda')
            simsiam_loss = torch.tensor(0., device='cuda')
            id_loss = torch.tensor(0., device='cuda')
            mae_loss = torch.tensor(0., device='cuda')

            if cfg.MODEL.HEAD.SDM_LOSS:
                sdm_loss = compute_sdm(image_feats, text_feats, pid.cuda(), logit_scale)  

            if cfg.MODEL.HEAD.ID_LOSS:
                id_loss = compute_id(image_logits, text_logits, pid.cuda())

            if cfg.MODEL.HEAD.ENABLE_MLM:
                mlm_scores = mlm_scores.float().reshape(-1, cfg.MODEL.VOCAB_SIZE)
                mlm_labels = mlm_labels.reshape(-1).cuda()
                mlm_loss = compute_mlm(mlm_scores, mlm_labels)
                
                pred = mlm_scores.max(1)[1]
                mlm_label_idx = torch.nonzero(mlm_labels)
                mlm_acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
                mlm_accuracy.update(mlm_acc.item(), image.size(0))

            if cfg.MODEL.HEAD.ENABLE_MAE:
                target, mae_pred, mask = output['mae']
                mae_loss = (mae_pred - target) ** 2
                mae_loss = mae_loss.mean(dim = -1)
                mae_loss = (mae_loss*mask).sum() / mask.sum()

            loss = sdm_loss + simsiam_loss + id_loss + mlm_loss + mae_loss
            optimizer.zero_grad()

            losses.update(loss.item(), image.size(0))
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step % (len(trainloader) * 30))

            if wandb is not None:
                wandb.log({'Loss': loss, 
                           'SDM Loss': sdm_loss, 'SimSiamLoss': simsiam_loss, 
                           'ID Loss': id_loss, "mlm_loss": mlm_loss,
                            "mlm_acc": mlm_acc, "MAE LOSS" : mae_loss,  
                           'lr': optimizer.param_groups[0]['lr']})
    return losses.avg, global_step, (rank1, rank5, rank10, mAP)
