import torch, os, datetime
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time


import transforms as transforms
from coco_utils import CocoDetection
from coco_utils import ConvertCocoPolysToMask
from coco_utils import resize, resizeVal, normalize
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import util, math

import torchvision, copy
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



obj_det_ave_loss = 0.0

def get_instance_of_faster_rcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 16)
    resnet = model.backbone.body


    ###############################################################
    #    FasterRCNN ResNet backbone uses FrozenBatchNorm layers   #
    #                                                             #
    #    Replace them with nn.BatchNorm2d layers.                 #
    ###############################################################
    bn_to_replace = []
    for name, module in resnet.named_modules():
        if isinstance(module, torchvision.ops.misc.FrozenBatchNorm2d):
            print('adding ', name)
            bn_to_replace.append(name)

    # Iterate all layers to change
    for layer_name in bn_to_replace:
        # Check if name is nested
        *parent, child = layer_name.split('.')
        # Nested
        if len(parent) > 0:
            # Get parent modules
            m = resnet.__getattr__(parent[0])
            for p in parent[1:]:
                m = m.__getattr__(p)
            # Get the FrozenBN layer
            orig_layer = m.__getattr__(child)
        else:
            m = resnet.__getattr__(child)
            orig_layer = copy.deepcopy(m) # deepcopy, otherwise you'll get an infinite recusrsion
        # Add your layer here
        in_channels = orig_layer.weight.shape[0]
        bn = torch.nn.BatchNorm2d(in_channels)
        with torch.no_grad():
            bn.weight = torch.nn.Parameter(orig_layer.weight)
            bn.bias = torch.nn.Parameter(orig_layer.bias)
            bn.running_mean = orig_layer.running_mean
            bn.running_var = orig_layer.running_var
        m.__setattr__(child, bn)

    return model


def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train_with_two(net, obj_det_net, lane_detection_data_loader, lane_loss_dict, lane_optimizer, lane_scheduler, logger, epoch, lane_metric_dict, use_aux, 
                        obj_detection_data_loader, obj_optimizer, obj_scheduler):
    global obj_det_ave_loss
    net.train()

    cnt_lane = len(lane_detection_data_loader)
    cnt_obj  = len(obj_detection_data_loader)

    lane_progress_bar = dist_tqdm(lane_detection_data_loader,)
    obj_progress_bar  = dist_tqdm(obj_detection_data_loader,)
    t_data_0 = time.time()

    #lane_det_itr = iter(lane_detection_data_loader)
    lane_det_itr = iter(lane_progress_bar)
    obj_det_itr  = iter(obj_progress_bar)

    b_lane_idx = 0
    b_obj_idx  = 0
    obj_data_available = True
    lane_data_available = True 
    while obj_data_available:
        ########################################
        #          Lane Detection              #
        ########################################
        try:
            run_times = 2 if cnt_lane>cnt_obj else 1


            for i in range(run_times):
                data_label = next(lane_det_itr)
                t_data_1 = time.time()
                reset_metrics(lane_metric_dict)
                global_step = epoch * len(lane_detection_data_loader) + b_lane_idx
    
                t_net_0 = time.time()
                results = inference(net, data_label, use_aux)
    
                loss = calc_loss(lane_loss_dict, results, logger, global_step)
                lane_optimizer.zero_grad()
                loss.backward()
                lane_optimizer.step()
                lane_scheduler.step(global_step)
                t_net_1 = time.time()
    
                results = resolve_val_data(results, use_aux)
    
                update_metrics(lane_metric_dict, results)
                if global_step % 20 == 0:
                    for me_name, me_op in zip(lane_metric_dict['name'], lane_metric_dict['op']):
                        logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
                logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
    
                if hasattr(lane_progress_bar,'set_postfix'):
                    kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(lane_metric_dict['name'], lane_metric_dict['op'])}
                    lane_progress_bar.set_postfix(loss = '%.3f' % float(loss),
                                            data_time = '%.3f' % float(t_data_1 - t_data_0),
                                            net_time = '%.3f' % float(t_net_1 - t_net_0),
                                            **kwargs)
                t_data_0 = time.time()
                b_lane_idx += 1

            cnt_lane -= run_times

        except StopIteration:
            lane_data_available = False  # 
            print('all lane obj detect samples iterated')
            del lane_det_itr
            break


        ########################################
        #        Object Detection              #
        ########################################
        try:
            #run_times = min(10,cnt_obj) if cnt_obj>cnt_lane else 1

            #for i in range(run_times):
            images, targets = next(obj_det_itr)
            t_data_1 = time.time()

            global_step = epoch * len(obj_detection_data_loader) + b_obj_idx

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            t_net_0 = time.time()
            loss_dict = obj_det_net(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            obj_det_ave_loss = (obj_det_ave_loss*global_step + float(losses) )/(global_step+1)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = util.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            obj_optimizer.zero_grad()
            losses.backward()
            obj_optimizer.step()

            obj_scheduler.step()
            t_net_1 = time.time()

            if hasattr(obj_progress_bar,'set_postfix'):
                # kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(lane_metric_dict['name'], lane_metric_dict['op'])}
                obj_progress_bar.set_postfix(loss = '%.3f' % obj_det_ave_loss, cur_loss= '%.3f' % float(losses),
                                        data_time = '%.3f' % float(t_data_1 - t_data_0),
                                        net_time = '%.3f' % float(t_net_1 - t_net_0))
                                        #**kwargs)

            t_data_0 = time.time()
            b_obj_idx += 1

            #cnt_obj -=run_times
            cnt_obj -=1
            # print(f"[Faster RCNN] Loss {loss_value}")
            
        except StopIteration:
            obj_data_available = False  # 
            print('all lane obj detect samples iterated')
            del obj_det_itr
            break



        #print(f"current idx: {b_idx}")
        #if b_idx == 10:
        #    break


def train(net, lane_detection_data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_aux):
    net.train()
    progress_bar = dist_tqdm(lane_detection_data_loader,)
    t_data_0 = time.time()


    for b_idx, data_label in enumerate(progress_bar):
    # for b_idx, data_label in enumerate(lane_detection_data_loader):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(lane_detection_data_loader,) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()
        


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = util.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    cnt = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #targets = [{k: v for k, v in t.items()} for t in targets]

        #print(f"debug: img shape: {len(images)}   {images[0].shape}")

        #for t in targets:
        #    print(f"debug:      {t}")
        #targets = [{k: v.to(device) for k, v in t} for list_target in targets]

        loss_dict = model(images, targets)

        #if cnt==10:
        #    break;
        # print("======== beg ========")
        # print(loss_dict)
        # print(images)
        # print(targets)
        # print("======== end ========")

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = util.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cnt+=1



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']


    lane_detection_train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, distributed, cfg.num_lanes)
    net = parsingNet(pretrained = True, backbone='50',cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_aux=cfg.use_aux).cuda()
    obj_det_net = net.faster_rcnn

    if distributed:
        net         = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank], find_unused_parameters=True)
        #net._set_static_graph()

    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0



    scheduler = get_scheduler(optimizer, cfg, len(lane_detection_train_loader))
    dist_print(len(lane_detection_train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)


    ###########################################################
    #                 Object Dectection                       #
    ###########################################################
    object_detection_train_data_path = "/nfs/home/data/Euclid/Dataset/ObjectDetection/BDD100K/bdd100k/images/100k/train/"

    transform_train = transforms.Compose([ConvertCocoPolysToMask(),
                                          transforms.ToTensor(),
                                          normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          #transforms.RandomHorizontalFlip(0.5),
                                          #resize(480,640)])
                                          resize(480,640)])


    # our dataset has background and the other 15 classes of the target objects (1+15)
    batch_size  = 8
    num_classes = 16
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    learning_rate = 0.001

    # define training and validation data loaders
    bdd100k_train                     = CocoDetection(object_detection_train_data_path, "./det_train_coco_gyr_dss.json", transforms=transform_train)

    if distributed:
        bdd100k_train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=bdd100k_train,shuffle=True,seed=42)
    else:
        bdd100k_train_sampler = torch.utils.data.RandomSampler(bdd100k_train)

    # bdd100k_train_sampler             =  DistributedSampler(dataset=bdd100k_train)
    object_detection_train_dataloader = DataLoader(bdd100k_train, batch_size=batch_size, shuffle=False, sampler=bdd100k_train_sampler, num_workers=8, collate_fn=util.collate_fn)
    # construct an optimizer
    params = [p for p in obj_det_net.parameters() if p.requires_grad]
    object_detection_optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(object_detection_optimizer, T_max=100)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(object_detection_optimizer, milestones=[25,38], gamma=0.1)



    for epoch in range(resume_epoch, cfg.epoch):

        train_with_two(net, obj_det_net, lane_detection_train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_aux, object_detection_train_dataloader, object_detection_optimizer, lr_scheduler)
        #train(net, lane_detection_train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_aux)
        #train_one_epoch(net.faster_rcnn, object_detection_optimizer, object_detection_train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        #lr_scheduler.step()

        
        save_model(net, optimizer, epoch ,work_dir, distributed)
    logger.close()
