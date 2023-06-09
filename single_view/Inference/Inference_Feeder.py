import os
import torch.utils.data as data
import torchvision.io as io
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import multiprocessing
import matplotlib.pyplot as plt
def generate_figure(data, save_path=0):

    data = data.astype(np.uint8)

    # 将数据从(3, 360, 384)转换为(360, 384, 3)
    data = np.transpose(data, (1, 2, 0))
    plt.imshow(data)
    plt.show()
    # 将数据保存为图像文件
    plt.imsave(f"figure/image_{save_path}.png", data)
from collections import Counter
class Feeder(data.Dataset):
    def __init__(self, videos_dir):
        self.videos_dir = videos_dir
        self.videos = []
        self.labels = []
        self.labels = torch.zeros(1091)
    def _load_label_numpy(self):
        self.labels = torch.from_numpy(np.load(self.labels_dir, allow_pickle=True))
        print(self.labels)
        print(len(self.labels))

    def __getitem__(self, index):
        video_path = os.path.join(self.videos_dir, f"video_{index}.npy")
        # print(video_path)
        frames = torch.from_numpy(np.load(video_path, allow_pickle=True))
        # print(frames.dtype)
        label = self.labels[index].long()
        # print(label.dtype)

        return frames, label

    def __len__(self):
        return len(self.labels)


def feeder_data_generator(dataset, batch_size):

    data_loader = torch.utils.data.DataLoader(dataset, shuffle= True,batch_size=batch_size,
                                                  num_workers=2,
                                                  pin_memory=True)

    return data_loader



if __name__ == '__main__':
    # multiprocessing.freeze_support()
    label_dir = "/public/home/wangchy5/CPR/R3d/TestData"
    train_dataset = Feeder(label_dir)
    train_data_loader = feeder_data_generator(train_dataset, batch_size=1)
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0

    for batch_idx, (data, target) in enumerate(train_data_loader):
        print(data.shape)
        print(target)
        # if batch_idx ==0:
        #
        #     print(data.dtype)
        #     # data = torch.from_numpy(data)
        #     data =data.permute(0,2,1,3,4)
        #     print(data.shape)
        #     print(data.dtype)
        #     data =data.numpy()
        #     generate_figure(data[0,1,:,:,:])
        # if target == 0:
        #     count_0 += 1
        # elif target == 1:
        #     count_1 += 1
        # elif target == 2:
        #     count_2 += 1
        # elif target == 3:
        #     count_3 += 1
        # elif target == 4:
        #     count_4 += 1
        # print(count_0)
        # print(count_1)
        # print(count_2)
        # print(count_3)
        # print(count_4)


