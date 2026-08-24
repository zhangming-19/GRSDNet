import os
import torch
import yaml

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as fp:  # contralearn/    './config.yaml'
    param = yaml.safe_load(fp)

# data_info
data_info = param['data_info']
data_dir = data_info['data_dir']
wav_dir = data_info['wav_dir']
raw_wav_dir = data_info['raw_wav_dir']
model_dir = data_info['model_dir']
feat_dir = data_info['feat_dir']
n_mels = data_info['n_mels']
n_fft = data_info['n_fft']
hop_hength = data_info['hop_hength']
mach_index = data_info['mach_index']

# man_control
man_control = param['man_control']
data_names = man_control['data_names']
model_name = man_control['model_name']
cur_dataset = man_control['cur_dataset']
loss_choice = man_control['loss_choice']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = man_control['cuda']
torch.cuda.device(cuda)  # torch.cuda.set_device(1)
mode = man_control['mode']
train_pickle = man_control['train_pickle']
valid_pickle = man_control['valid_pickle']
test_pickle = man_control['test_pickle']
data_name = man_control['data_name']
fs = man_control['fs']
wav_time = man_control['wav_time']
wav_time_frames = man_control['wav_time_frames']
leaf_batch = man_control['leaf_batch']
pre_batch = man_control['pre_batch']
ext_mode = man_control['ext_mode']
da_multi = man_control['da_multi']
da = man_control['da']
test_multi = man_control['test_multi']
premodel_ext = man_control['premodel_ext']
zero_normal = man_control['zero_normal']
feat_normal = man_control['feat_normal']
# train
train = param['train']
emb_dim = train['emb_dim']
input_dim = train['input_dim']
seed = train['seed']
batch_size = train['batch_size']
MAX_EPOCH = train['epoch']
LR = train['lr']
weight_decay = train['weight_decay']
decimal_count = train['decimal_count']
loss_thre = train['loss_thre']
num_workers = train['num_workers']
class_num = train['class_num']

# da
da = param['da']
# Shift Audio
shift_min = da['shift_min']
shift_max = da['shift_max']
# Change Tempo
tempo_min = da['tempo_min']
tempo_max = da['tempo_max']
tempo_stride = da['tempo_stride']
# Add noise
noise_min_factor = da['noise_min_factor']
noise_max_factor = da['noise_max_factor']
noise_stride = da['noise_stride']
noise_time_min = da['noise_time_min']
noise_time_max = da['noise_time_max']
# Pitch Shifting
pitch_min = da['pitch_min']
pitch_max = da['pitch_max']
pitch_stride = da['pitch_stride']
