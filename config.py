import os
import torch
import yaml

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open('config.yaml', 'r', encoding='utf-8') as fp:  # 'config.yaml'
    param = yaml.safe_load(fp)

# data_info
data_info = param['data_info']
data_dir = data_info['data_dir']
wav_dir = data_info['wav_dir']
pth_dir = data_info['pth_dir']
feat_dir = data_info['feat_dir']
result_dir = data_info['result_dir']
fs = data_info['fs']
n_mels = data_info['n_mels']
n_fft = data_info['n_fft']
hop_length = data_info['hop_length']
tar_time = data_info['tar_time']
mach_index = param['mach_index']

# man_control
feat = param['feat']
zero_normal = feat['zero_normal']
feat_normal = feat['feat_normal']
cur_domain = feat['cur_domain']
secID = feat['secID']

# train
train = param['train']
# control
cuda = train['cuda']
torch.cuda.set_device(cuda)  # torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = train['mode']
feat_path = train['feat_path']
train_joblib = train['train_joblib']
test_joblib = train['test_joblib']
learn_method = train['learn_method']
model_name = train['model_name']
data_name = train['data_name']
mixup_choice = train['mixup_choice']
mixup_lossweight = train['mixup_lossweight']
choose_model_nums = train['choose_model_nums']
batch_size = train['batch_size']
MAX_EPOCH = train['epoch']
LR = train['lr']
weight_decay = train['weight_decay']
premodel_ext = train['premodel_ext']
kmean_needtrain = train['kmean_needtrain']
kmeans_needtest = train['kmeans_needtest']
kmean_draw3D = train['kmean_draw3D']
# AER参数
cnn_dims = train['cnn_dims']
global_pooling = train['global_pooling']
# mlp
head_num = train['head_num']
head_size = train['head_size']
mlp_hidden_size = train['mlp_hidden_size']
low_dim = train['low_dim']
# contra loss
temp = train['temp']
contra_loss_weight = train['contra_loss_weight']
# stable
input_dim = train['input_dim']
seed = train['seed']
threshold_count = train['threshold_count']
decimal_count = train['decimal_count']
loss_thre = train['loss_thre']
threshold_alpha = train['threshold_alpha']
pauc_value = train['pauc_value']
class_num = train['class_num']
detect_num = train['detect_num']
num_workers = train['num_workers']
alpha = train['alpha']

# kmeans
kmeans_model = param['kmeans_model']
# RandTimeShift
max_iter = kmeans_model['max_iter']
tol = kmeans_model['tol']
