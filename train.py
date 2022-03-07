from easydict import EasyDict as edict
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import model.crnn as crnn
import utils.utils as utils
from Dataset import dataset
import function
from Dataset import alphabets as alphabets
from tensorboardX import SummaryWriter

def main():
    #create output folder
    output_dict = utils.create_log_folder()

    #cudnn
    """Search the most suitable convolution implementation algorithm for each convolutional layer of the entire network,
     and then realize the acceleration of the network. The applicable scenario is that the network structure is fixed 
     (not dynamically changing), and the input shape of the network (including batch size, image size, and input channel)
      is unchanged. In fact, it is more applicable in general."""
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    # writer dict
    writer_dict = {
        'writer':SummaryWriter(log_dir = output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0
    }

    # construct face relatede
    model = crnn.get_crnn(num_classes = 0)

    #get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    #define loss function
    criterion = torch.nn.CTCLoss()

    begin_epoch = 0
    lr_step = [60, 80]
    lr_factor = 0.1
    is_finetune = True
    finetune_checkpoint = 'output/checkpoints/mixed_second_finetune_acc_97P7.pth'


    last_epoch = begin_epoch
    optimizer = utils.get_optimizer(model)

    if isinstance(lr_step, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, lr_step, lr_factor, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, lr_step, lr_factor, last_epoch - 1
        )

    if is_finetune:
        model_state_file = finetune_checkpoint
        pass