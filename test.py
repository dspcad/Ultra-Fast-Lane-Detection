import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch


import transforms as transforms
from coco_utils import CocoDetection
from coco_utils import ConvertCocoPolysToMask
from coco_utils import resize, resizeVal
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import util, math
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import time



def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = util.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    #iou_types = _get_iou_types(model)
    iou_types = ["bbox"]

    evaluator_15c = []
    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.coco_eval["bbox"].params.catIds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    evaluator_15c.append(coco_evaluator)

    for i in range(1,16):
        coco_evaluator = CocoEvaluator(coco, iou_types)
        coco_evaluator.coco_eval["bbox"].params.catIds = [i]
        evaluator_15c.append(coco_evaluator)


    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()

        for coco_evaluator in evaluator_15c:
            coco_evaluator.update(res)

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats {i}:", metric_logger)
    for coco_evaluator in evaluator_15c:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    #cat_id2name = {1: 'pedestrian', 2: 'rider', 3: 'car', 4: 'truck', 5 : 'bus', 6 : 'motorcycle', 7 : 'bicycle', 8 : 'traffic light', 9 : 'traffic sign', 10 : 'green', 11 : 'yellow', 12 : 'red', 13 : 'do not enter', 14 : 'stop sign', 15 : 'speed limit'}
    #for catId in coco_evaluator.coco_gt.getCatIds():
    #    print(f"catId: {catId}")
    #    print(f"catId: {cat_id2name[catId]} ({catId})")
    #    cocoEval = copy.deepcopy(coco_evaluator)
    #    cocoEval.coco_eval["bbox"].params.catIds = [catId]
    #    cocoEval.evaluate()
    #    cocoEval.accumulate()
    #    cocoEval.summarize()
    cat_id2name = {0: '15 Classes', 1: 'pedestrian', 2: 'rider', 3: 'car', 4: 'truck', 5 : 'bus', 6 : 'motorcycle', 7 : 'bicycle', 8 : 'traffic light', 9 : 'traffic sign', 10 : 'green', 11 : 'yellow', 12 : 'red', 13 : 'do not enter', 14 : 'stop sign', 15 : 'speed limit'}
    for i, coco_evaluator in enumerate(evaluator_15c):
        print(f"Category: {cat_id2name[i]}")
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
    return evaluator_15c


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = torch.load(cfg.test_model)
    #net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),
    #                use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    #state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    #compatible_state_dict = {}
    #for k, v in state_dict.items():
    #    #print(f"debug: k: {k} --> {k[7:]}")
    #    #if 'module.' in k:
    #    #    compatible_state_dict[k[7:]] = v
    #    if 'model.' in k:
    #        print(f"debug: k: {k} --> {k[6:]}")
    #        compatible_state_dict[k[6:]] = v
    #    else:
    #        #print(f"debug: k: {k}")
    #        compatible_state_dict[k] = v

    #net.load_state_dict(compatible_state_dict, strict = False)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    ######################################
    #          Object Detection          #
    ######################################
    val_data_path   = "/NFS/share/Euclid/Dataset/ObjectDetection/BDD100K/bdd100k/images/100k/val/"

    transform_val   = transforms.Compose([ConvertCocoPolysToMask(),
                                          transforms.ToTensor(),
                                          resizeVal(480,640)])

    bdd100k_val      = CocoDetection(val_data_path, "./det_val_coco_gyr_dss.json", transforms=transform_val)
    val_dataloader   = DataLoader(bdd100k_val, batch_size=8, shuffle=False, num_workers=8, collate_fn=util.collate_fn)

    
    model = net.faster_rcnn
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        img, _ = bdd100k_val[0]
        evaluate(model, val_dataloader, device=device)
        prediction = model([img.to(device)])


        print(prediction[0]['boxes'])
        img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        img1.save("target.png")

        img1 = torchvision.transforms.ToTensor()(img1)
        img1 = torchvision.transforms.ConvertImageDtype(dtype=torch.uint8) (img1)
        colors=["yellow" for i in prediction[0]['boxes']]
        img1 = torchvision.utils.draw_bounding_boxes(img1, prediction[0]['boxes'], colors=colors ,width=3,fill=True)
        target = Image.fromarray(img1.permute(1,2,0).byte().numpy())
        target.save("target1.png")


    eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, False, distributed)
