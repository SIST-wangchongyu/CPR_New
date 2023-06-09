import os
import torch.utils.data as data
import torchvision.io as io
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

import random
from collections import Counter
class Feeder(data.Dataset):
    def __init__(self, videos_dir, labels_dir,flip):
        # self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.videos_dir = videos_dir
        self.videos = []
        self.labels = []
        self._load_label_numpy()
        self.class_sample_count = np.unique(self.labels, return_counts=True)[1]
        self.weight = 1. / self.class_sample_count
        self.samples_weight = self.weight[self.labels]
        self.reweighting = [0,0,0,0,0]
        self.count_weight()
        self.flip =flip

        self.sampler = WeightedRandomSampler(self.samples_weight, len(self.labels))

        # self.max_frame =224
        # self.valid_frame_dict = {}
    def count_weight(self):
        for i in self.labels:
            self.reweighting[int(i)]+=1
    def _load_label_numpy(self):
        self.labels = torch.from_numpy(np.load(self.labels_dir, allow_pickle=True))


    def __getitem__(self, index):
        video_path = os.path.join(self.videos_dir, f"video_{index}.npy")

        frames = np.load(video_path, allow_pickle=True)
        if self.flip ==1 and random.random() <0.5:
            frames = np.flip(frames, axis=(3))

        label = self.labels[index].long()
        # print(label.dtype)

        return torch.from_numpy(frames.copy()), label

    def __len__(self):
        return len(self.labels)


def feeder_data_generator(dataset, batch_size, sampler):
    if sampler == 1:
        data_loader = torch.utils.data.DataLoader(dataset, sampler=dataset.sampler, batch_size=batch_size,
                                                  num_workers=0,
                                                  pin_memory=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                                  num_workers=0,
                                                  pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                           pin_memory=True,sampler = dataset.sampler, shuffle=False)
    return data_loader



if __name__ == '__main__':
    # multiprocessing.freeze_support()
    root = 'F:\\CPR\\CPR_git\\single_view'
    train_label_root = os.path.join(root, '../../CPR_6/labels', 'labels_16frames_train_without_A_crop_single_view.npy')
    train_data_root = os.path.join(root, 'Video_16frames_train_without_A_crop_singleview')

    train_dataset = Feeder(train_data_root,train_label_root,flip=0)
    train_data_loader = feeder_data_generator(train_dataset, batch_size=1,sampler=1)
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0

    for batch_idx, (data, target) in enumerate(train_data_loader):
        print(data.shape)
        print(target)

