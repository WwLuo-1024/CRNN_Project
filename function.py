from __future__ import absolute_import
import time
import utils.utils as utils
import torch
class AverageMeter(object):
    """Computng and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict = None, output_dict = None):
    print_freq = 10
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, idx) in enumerate(train_loader):
        #measure data time
        data_time.update(time.time() - end)
        labels = utils.get_batch_label(dataset, idx)
        input = input.to(device)

        #inference
        preds = model(input).cpu()

        #compute loss
        batch_size = input.size(0)
        text, length = converter.encode(labels) # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * bat
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(time.time() - end)

        batch_time.update(time.time() - end)
        if i % print_freq == 0:
            msg = 'Epoch : [{0}][{1}/{2}]\t' \
                  'Time : {batch_time.val:3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed : {speed:.1f}samples/s\t' \
                  'Data : {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss : {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time = batch_time,
                speed = input.size(0) / batch_time.val,
                data_time = data_time, loss = losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()

