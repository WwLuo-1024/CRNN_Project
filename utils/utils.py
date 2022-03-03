import torch.optim as optim
import time
from pathlib import Path
import torch

def get_optimizer(model):

    optimimzer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = 0.0001
    )

    return optimimzer

def create_log_folder(dataset):
    root_out_dir = Path('output')
    #set up logger
    if not root_out_dir.exists():
        print('=> creating {}'.format(root_out_dir))
        root_out_dir.mkdir()

    dataset = dataset
    model = 'crnn'
    time_str = time.struct_time('%Y-%m-%d-%H-%M')
    checkpoints_output_dir = root_out_dir / dataset / model / time_str / 'checkpoints'

    print('=> creating{}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents = True, exist_ok = True)

    tensorboard_log_dir = root_out_dir / dataset / model / time_str / 'log'
    print('=> creating{}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents = True, exist_ok = True)

    return {'chs_dir':str (checkpoints_output_dir), 'tb_dir':str(checkpoints_output_dir)}

def get_batch_label(dataset, index):
    label = []
    for idx in index:
        label.append(list(dataset.labels[idx].values())[0])
    return label

class strLabelConverter(object):
    pass