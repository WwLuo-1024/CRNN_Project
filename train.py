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
from collections import OrderedDict

def main():
    begin_epoch = 0
    lr_step = [60, 80]
    lr_factor = 0.1
    is_finetune = True
    finetune_freeze = True
    finetune_checkpoint = 'output/checkpoints/mixed_second_finetune_acc_97P7.pth'
    is_resume = False
    resume_file = ''
    train_batch_size_per_gpu = 32
    test_batch_size_per_gpu = 16
    num_worker = 1
    train_end_epoch = 100

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
        if model_state_file == '':
            print("--no checkpoint found--")
        checkpoint = torch.load(model_state_file, map_location = 'cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v

        model.cnn.load_state_dict(model_dict)
        if finetune_freeze:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif is_resume: #Restoring model parameters from interrupted training
        model_state_file = resume_file
        if model_state_file == '':
            print("--no checkpoint found--")
        checkpoint = torch.load(model_state_file, map_location = 'cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)

    utils.model_info(model)
    train_dataset = dataset._360CC(is_train = True)
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = train_batch_size_per_gpu,
        shuffle = True,
        num_workers = num_worker,
        pin_memory = False,
    )
    val_dataset = dataset._360CC(is_train = False)
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = test_batch_size_per_gpu,
        shuffle = True,
        num_workers = num_worker,
        pin_memory = False,
    )

    best_acc = 0.5
    converter = utils.strLabelConverter(alphabets.alphabet)
    for epoch in range(last_epoch, train_end_epoch):
        function.train(train_loader, train_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()
        acc = function.validate(val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best", is_best)
        print("best acc is:", best_acc)
        #save checkpoint
        torch.save(
            {
                "state_dict":model.state_dict(),
                "epoch":epoch + 1,
                "best_acc":best_acc,
            }, os.path.join(output_dict['chs_dir'],"checkpoint_{}_acc_{:.4f}.pth".format(epoch,acc))
        )
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()