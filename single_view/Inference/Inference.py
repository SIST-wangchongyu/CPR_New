import os.path

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import sys
import torch.nn.parallel
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
sys.path.append("../../")
from tensorboardX import SummaryWriter
import numpy as np
import argparse
from tqdm import tqdm
from Inference_Feeder import Feeder,feeder_data_generator
from ../ResNet3D import New_R3d

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size (default: 4)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit (default: 40)')

parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=10,metavar='N',
                    help='report interval (default:10')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


root = 'F:\\CPR\\CPR_git\\single_view'


epochs =20
lr = 2e-3
batch_size = 1
n_classes = 5
kernel_size = 2
channel_sizes = 128
steps = 0
print(args)

#optical flow
# train_label_root= '/public/home/wangchy5/CPR/R3d/labels/labels_8frames_train_without_A_crop.npy'
# train_data_root = '/public/home/wangchy5/CPR/R3d/Video_15frames_optical_flow_train'
# test_data_root ='/public/home/wangchy5/CPR/R3d/Video_15frames_optical_flow_test'
# test_label_root ='/public/home/wangchy5/CPR/R3d/labels/labels_8frames_test_without_A_crop.npy'
#video
inference_data_root =os.path.join(root,'Inference_Data')
inference_dataset  = Feeder(inference_data_root)
inference_data_loader = feeder_data_generator(inference_dataset,batch_size=batch_size)
device_count = torch.cuda.device_count()
device_ids = list(range(device_count))
print(device_ids)
model = New_R3d()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(model)
model.to(device)

def inference(epoch):
    model.eval()
    process = tqdm(inference_data_loader)
    counter =0

    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(process):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            target_cpu = target.cpu()
            pred_cpu = pred.cpu()
            output_cpu =output.cpu()
            top_k_values, top_3 = torch.topk(output, k=3)

            sum_out=0
            for i in range(5):
                sum_out+= np.exp(output_cpu[0][i])
            tensor = torch.Tensor([[1, 2, 3]])
            for i in range(3):
                tensor[0][i] = float(np.exp(output_cpu[0][top_3[0][i]])/sum_out)

            predict_action = top_3
            confidence_score =tensor
            if counter ==0:
                tensor_for_action = predict_action
                tensor_for_score = confidence_score
                counter=1
            else:
                tensor_for_action = torch.cat((tensor_for_action,predict_action),dim=0)
                tensor_for_score = torch.cat((tensor_for_score, confidence_score), dim=0)


        tensor_for_score =torch.round(tensor_for_score*1000)/1000
        confidence_score=tensor_for_score.numpy()
        top_3_action=tensor_for_action.cpu().numpy()
        print(confidence_score.shape)
        print(top_3_action.shape)
        np.save("confidence_score",confidence_score)
        np.save("top_3_action",top_3_action)
        print(confidence_score)
        print( top_3_action)







if __name__ == "__main__":
    root = 'F:\\CPR\\CPR_git\\single_view'
    path = os.path.join(root,'weight','R3d_3fc_finetune_30epoch_26_th_0.002_10_5')
    checkpoint  = torch.load(path)
    model.load_state_dict(checkpoint)
    inference(1)