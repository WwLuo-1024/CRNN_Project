import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    pass

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn = 2, laekyRelu = False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2] #kernal size
        ps = [1, 1, 1, 1, 1, 1, 0] #padding size
        ss = [1, 1, 1, 1, 1, 1, 1] #strides size
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization = False):
            input_channels = nc if i == 0 else nm[i - 1]
            out_channels = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(input_channels, out_channels, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(out_channels))
            if laekyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace = True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0) #1x32xW (W:orginal image width)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2)) #64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2)) #128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2),(2, 1), (0, 1))) #256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1))) #512x2x16
        convRelu(6, True) # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), # nh:number of hidden units
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        #conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        print(conv.size())
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2) # batchx512xwidth
        conv = conv.permute(2, 0, 1) #[w, b, c]
        output = F.log_softmax(self.rnn(conv), div = 2)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_crnn(num_classes):
    model = CRNN(280, 1, num_classes + 1, 256)
    model.apply(weights_init)

    return model