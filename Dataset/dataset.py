from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2

class _360CC(data.Dataset):
    def __init__(self, is_train = True):
        self.root = "./data/image"
        self.is_trian = is_train
        self.input_height = 32
        self.input_width = 160 #resized width
        self.dataset_name = "360CC"
        self.mean = np.array(0.588, dtype = np.flor32)
        self.std = np.array(0.193, dtype = np.float32)

        char_file = "./data/text/char_std_5990.txt"  # data dictionary
        with open(char_file, 'rb') as file:
            char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
        if is_train == True:
            txt_file = "./data/text/train.txt"
        else:
            txt_file = "./data/text/test.txt"

        self.labels = []
        with open(txt_file, 'r', encoding = 'utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imageName = c.split(' ')[0]
                indices = c.split(' ')[1:]
                string = ''.join([char_file[int(idx)] for idx in indices])
                self.labels.append({imageName: string})
        print("load {} images~!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item, idx):
        image_name = list(self.labels[idx].keys())[0]
        image = cv2.imread(os.path.join(self.root, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_h, image_w = image.shape
        #data preprocessing
        image = cv2.resize(image, (0,0), fx=self.input_width/image_w, fy = self.input_height / image_h) #cv2.resize(img, (0, 0), fx=2, fy=1) 功能：使得图像x轴变化为原来的2倍，y轴不变
        image = np.reshape(image, (self.input_height, self.input_width))
        image = image.astype(np.float32)
        image = (image/255. - self.mean) / self.std
        image = image.transpose([2,0,1])
        return image, idx

if __name__ == '__main__':
    pass
    ##测试打印字典
    # char_file = "./data/text/char_std_5990.txt"  # data dictionary
    # with open(char_file, 'rb') as file:
    #     char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
    # print(char_dict)

    ##测试打印数据每行文字内容
    # labels = []
    # txt_file = "./data/text/test.txt"
    # with open(txt_file, 'r', encoding='utf-8') as file:
    #     contents = file.readlines()
    #     for c in contents:
    #         imageName = c.split(' ')[0]
    #         indices = c.split(' ')[1:]
    #         string = ''.join([char_dict[int(idx)] for idx in indices])
    #         labels.append({imageName:string})
        # print(labels)

    ##测试打印数据预处理结果
    # root = "./data/image"
    # image_name = list(labels[0].keys())[0]
    # image = cv2.imread(os.path.join(root, image_name))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # image_h, image_w = image.shape
    # #data preprocessing
    # image = cv2.resize(image, (0,0), fx = 160/image_w, fy = 32 / image_h) #cv2.resize(img, (0, 0), fx=2, fy=1) 功能：使得图像x轴变化为原来的2倍，y轴不变
    # image = np.reshape(image, (160, 32, 1))
    # image = image.astype(np.float32)
    # image = (image/255. - 0.588) / 0.193
    # image = image.transpose([2,0,1])
    # print(image)