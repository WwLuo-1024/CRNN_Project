import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import utils.utils as utils
import model.crnn as crnn
import Dataset.alphabets as alphabets
from easydict import EasyDict as edict

def recognition(img, model, converter, device):
    model_input_height = 32
    model_input_width = 160
    orginal_image_width = 280
    dataset_mean = 0.588
    dataset_std = 0.193

    h, w = img.shape
    # first step: resize the height and width of image to (32, xxx)
    img = cv2.resize(img, (0, 0), fx = model_input_height / h, fy = model_input_height / h, interpolation = cv2.INTER_CUBIC)

    #second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_wur = int(img.shape[1] / (orginal_image_width / model_input_width))
    img = cv2.resize(img, (0, 0), fx = w_wur / w, fy = 1.0, interpolation = cv2.INTER_CUBIC)
    img = np.reshape(img, (model_input_height, w_wur, 1))

    #normalize
    img = img.astype(np.float32)
    img = (img / 255. - dataset_mean) / dataset_std
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw = False)

    print('results: {0}'.format(sim_pred))

if __name__ == '__main__':
    checkpointUrl = 'output/360CC/crnn/2022-02-23-21-32/checkpoints/checkpoint_0_acc_0.9603.pth'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = crnn.get_crnn(len(alphabets.alphabet)).to(device)
    print('loading pretrained model from {0}'.format(checkpointUrl))
    checkpoint = torch.load(checkpointUrl)

    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()
    img = cv2.imread('test/003.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(alphabets.alphabet)

    recognition(img, model, converter, device)
    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))
