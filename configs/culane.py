# DATA
dataset='CULane'
data_root = '/nfs/home/data/Euclid/Dataset/LaneDetection/CULane/'

# TRAIN
epoch = 50
batch_size = 8
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.001
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '50'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/us000147/project/two_heads_bdd100k/Ultra-Fast-Lane-Detection/logs/'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4




