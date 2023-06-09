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
from Feeder import Feeder,feeder_data_generator
from ResNet3D import New_R3d

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

root = './data/mnist'

# seq_length = int(784 / input_channels) #Need to change



epochs =30
lr = 2e-4
batch_size = 8
n_classes = 5
kernel_size = 2
channel_sizes = 128
steps = 0
writer = SummaryWriter('/public/home/wangchy5/CPR/R3d/Summary/summary_finetune_5epoch_resampling_15epoch_2e-4')
print(args)

#optical flow
# train_label_root= '/public/home/wangchy5/CPR/R3d/labels/labels_8frames_train_without_A_crop.npy'
# train_data_root = '/public/home/wangchy5/CPR/R3d/Video_15frames_optical_flow_train'
# test_data_root ='/public/home/wangchy5/CPR/R3d/Video_15frames_optical_flow_test'
# test_label_root ='/public/home/wangchy5/CPR/R3d/labels/labels_8frames_test_without_A_crop.npy'

#video
train_label_root='/public/home/wangchy5/CPR/R3d/labels/labels_16frames_train_without_A_crop.npy'
train_data_root = '/public/home/wangchy5/CPR/R3d/Video_16frames_train_without_A_crop_r3d'
test_data_root = '/public/home/wangchy5/CPR/R3d/Video_16frames_test_without_A_crop_r3d'
test_label_root ='/public/home/wangchy5/CPR/R3d/labels/labels_16frames_test_without_A_crop.npy'

train_dataset = Feeder(train_data_root,train_label_root)
train_data_loader = feeder_data_generator(train_dataset,batch_size=batch_size,sampler=0)
train_data_loader_resampling = feeder_data_generator(train_dataset,batch_size=batch_size,sampler=1)

test_dataset  = Feeder(test_data_root,test_label_root)
test_data_loader = feeder_data_generator(test_dataset,batch_size=batch_size,sampler=0)
device_count = torch.cuda.device_count()
device_ids = list(range(device_count))
print(device_ids)
model = New_R3d()

params_to_update = []
params_not_to_update = []

# print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = nn.DataParallel(model,device_ids=device_ids)
print(model)
for name, param in model.named_parameters():

    if name.startswith('module.r3d.fc') or name.startswith('r3d.fc'):
        print(name)
        params_to_update.append(param)

    # else:
    #     param.requires_grad = False
    #     params_not_to_update.append(param)

# optimizer = optim.SGD(params=params_to_update, lr=lr)
# optimizer = nn.DataParallel(optimizer,device_ids=device_ids)
optimizer = optim.SGD(params = model.parameters(),lr=lr)
optimizer_for_resampling = optim.SGD(params = params_to_update,lr=lr)
model.to(device)
global test_loss_list
global train_loss_list
test_loss_list =[]
train_loss_list =[]
def train(ep,flag):
    global steps
    train_loss = 0
    model.train()
    correct = 0
    total =0
    counter=0
    tensor_for_target = None
    tensor_for_pred   = None
    total_train_loss = 0
    total_train_accuracy =0
    counter_for_iteration = 0
    counter_for_accuracy = 0


    if flag ==0:
        process = tqdm(train_data_loader)
        for batch_idx, (data, target) in enumerate(process):
            counter_for_iteration+=1
            data, target = data.to(device), target.to(device)
            # print(data.shape)
            # print(data.size())
            # data = data.view(-1, input_channels, seq_length) # batch_size,3,784
            # if args.permute:
            #     data = data[:, :, permute]
            output = model(data)
            # print(output.shape)

            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).sum().item()
            total += len(target)
            loss = F.cross_entropy(output, target).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            if counter == 0:
                tensor_for_target = target.cpu()
                tensor_for_pred = pred.cpu()
                counter = 1
            else:
                tensor_for_target = torch.cat((tensor_for_target, target.cpu()), dim=0)
                tensor_for_pred = torch.cat((tensor_for_pred, pred.cpu()), dim=0)
            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                acc = (100. * correct) / total
                total_train_accuracy += acc
                counter_for_accuracy += 1
                total_train_loss += train_loss

                # print(batch_size)
                # print(batch_idx)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.4f}'.format(
                    ep, batch_idx * batch_size, len(train_data_loader.dataset),
                        100. * batch_idx / len(train_data_loader), train_loss.item() / args.log_interval, acc))
                correct = 0
                total = 0
                train_loss = 0
    else:
        process_resampling = tqdm(train_data_loader_resampling)
        for batch_idx, (data, target) in enumerate(process_resampling):
            counter_for_iteration += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output.shape)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            total += len(target)
            loss = F.cross_entropy(output, target).cuda()
            optimizer_for_resampling.zero_grad()
            loss.backward()
            optimizer_for_resampling.step()
            train_loss += loss
            if counter == 0:
                tensor_for_target = target.cpu()
                tensor_for_pred = pred.cpu()
                counter = 1
            else:
                tensor_for_target = torch.cat((tensor_for_target, target.cpu()), dim=0)
                tensor_for_pred = torch.cat((tensor_for_pred, pred.cpu()), dim=0)
            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                acc = (100. * correct) / total
                total_train_accuracy += acc
                counter_for_accuracy += 1
                total_train_loss += train_loss

                # print(batch_size)
                # print(batch_idx)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.4f}'.format(
                    ep, batch_idx * batch_size, len(train_data_loader.dataset),
                        100. * batch_idx / len(train_data_loader), train_loss.item() / args.log_interval, acc))
                correct = 0
                total = 0
                train_loss = 0






    confusion_mat = confusion_matrix(tensor_for_target, tensor_for_pred)
    print(confusion_mat)
    print_accuracy(confusion_mat)
    total_train_loss /= counter_for_iteration
    print("Train loss is:{:.4f}".format(total_train_loss))
    train_tag = 'Training Loss'
    writer.add_scalars(main_tag='Loss', tag_scalar_dict={train_tag:total_train_loss}, global_step=ep)
    writer.add_scalars(main_tag='Accuracy', tag_scalar_dict={'Training Accuracy': (total_train_accuracy)/ counter_for_accuracy}, global_step=epoch)
    train_loss_list.append(total_train_loss.cpu())
    print("Train Ac is:{:.4f}".format(( total_train_accuracy) / counter_for_accuracy))

def confusion_matrix(true_labels, predicted_labels):
    # 获取类别数量
    num_classes = 5

    # 初始化混淆矩阵
    conf_matrix = np.zeros((num_classes, num_classes))
    conf_matrix = conf_matrix.astype(int)

    # 计算混淆矩阵
    for i in range(len(true_labels)):
        true_label = int(true_labels[i])
        predicted_label = int(predicted_labels[i])
        conf_matrix[true_label][predicted_label] += 1

    return conf_matrix
def print_accuracy(arr):
    label_0 = float(arr[0][0]/arr[0].sum())
    label_1 = float(arr[1][1] / arr[1].sum())
    label_2 = float(arr[2][2] / arr[2].sum())
    label_3 = float(arr[3][3] / arr[3].sum())
    label_4 = float(arr[4][4] / arr[4].sum())
    print("人工呼吸:{:.4f}".format(label_0), end="\n")
    print("心脏按压:{:.4f}".format(label_1), end="\n")
    print("检查脉搏和呼吸:{:.4f}".format(label_2), end="\n")
    print("解开衣服:{:.4f}".format(label_3), end="\n")
    print("轻拍双肩:{:.4f}".format(label_4), end="\n")



def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    process = tqdm(test_data_loader)
    tensor_for_target =None
    tensor_for_pred = None
    counter =0
    test_loss =0
    counter_for_iteration =0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(process):

            counter_for_iteration+=1
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            # for i in pred.eq(target.data.view_as(pred)).cpu():
            #     if i == True:
            #         print(i)

            # print(batch_idx)
            # print(target)
            # print(pred)
            # print()
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            total   += len(target)
            target_cpu = target.cpu()
            pred_cpu = pred.cpu()
            output_cpu =output.cpu()
            loss = F.cross_entropy(output, target).cuda()
            test_loss += loss
            if counter ==0:
                tensor_for_target = target_cpu
                tensor_for_pred = pred_cpu
                counter=1
            else:
                tensor_for_target = torch.cat((tensor_for_target,target_cpu),dim=0)
                tensor_for_pred = torch.cat((tensor_for_pred, pred_cpu), dim=0)

        confusion_mat = confusion_matrix(tensor_for_target, tensor_for_pred)
        print(confusion_mat)
        print_accuracy(confusion_mat)
        test_loss/=counter_for_iteration
        print("Test loss is:{:.4f}".format(test_loss))
        test_loss_list.append(test_loss.cpu())
        test_tag = 'Test Loss'
        writer.add_scalars(main_tag='Loss', tag_scalar_dict={test_tag:test_loss.cpu()}, global_step=epoch)
        writer.add_scalars(main_tag='Accuracy', tag_scalar_dict={'Test Accuracy':100.*correct/total }, global_step=epoch)


if __name__ == "__main__":
    epoch_flag = 5
    for epoch in range(1, epochs + 1):
        if epoch <= epoch_flag:
            train(epoch,0)
        elif epoch == epoch_flag+1:
            # for name, param in model.named_parameters():
            #     if name.startswith('module.r3d.fc') or name.startswith('r3d.fc'):
            #         print(name)
            #     else:
            #         param.requires_grad = False
            train(epoch, 1)
        else:
            train(epoch, 1)

        if epoch % 2 == 0:
            test(epoch)
        if epoch % 5 == 0 & epoch <= epoch_flag:
            lr /= 4
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch % 5 == 0 & epoch > epoch_flag:
            lr =2e-4
            for param_group in optimizer_for_resampling.param_groups:
                param_group['lr'] = lr
    torch.save(model.state_dict(), 'R3d_finetune_5epoch_resampling_15epoch_2e-4')
    # checkpoint  = torch.load('/public/home/wangchy5/CPR/R3d/R3d_30epochs_3fc_finetune')
    # # print(checkpoint)
    # model.load_state_dict(checkpoint)
    # test(1)