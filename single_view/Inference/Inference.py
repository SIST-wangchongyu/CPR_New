import os.path
import torch
import torch.nn.parallel
import sys
import torch.nn.parallel
# -*- coding: utf-8 -*-
sys.path.append("../../")
import numpy as np
import argparse
from tqdm import tqdm
from Inference_Feeder import Feeder,feeder_data_generator
from ResNet3D import New_R3d



def inference(epoch):
    model.eval()
    process = tqdm(inference_data_loader)
    counter =0

    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(process):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]

            output_cpu =output.cpu()
            top_k_values, top_1 = torch.topk(output, k=1)

            sum_out=0
            for i in range(5):
                sum_out+= np.exp(output_cpu[0][i])
            tensor = torch.Tensor([[1]])
            tensor[0] = float(np.exp(output_cpu[0][top_1[0][0]])/sum_out)

            predict_action = top_1
            confidence_score =tensor
            if counter ==0:
                tensor_for_action = predict_action
                tensor_for_score = confidence_score
                counter=1
            else:
                tensor_for_action = torch.cat((tensor_for_action,predict_action),dim=0)
                tensor_for_score = torch.cat((tensor_for_score, confidence_score), dim=0)


        tensor_for_score =torch.round(tensor_for_score*1000)/1000
        confidence_score =tensor_for_score.numpy()
        top_1_action     =tensor_for_action.cpu().numpy()
        np.save("confidence_score",confidence_score)
        np.save("top_1_action",top_1_action)
        print(confidence_score)
        print( top_1_action)







if __name__ == "__main__":
    root = 'F:\\CPR\\CPR_6\\single_view'
    batch_size = 1
    # inference video

    inference_data_root = os.path.join(root, 'Inference', 'Inference_Data')
    inference_dataset = Feeder(inference_data_root)
    inference_data_loader = feeder_data_generator(inference_dataset, batch_size=batch_size)
    device_count = torch.cuda.device_count()
    device_ids = list(range(device_count))

    model = New_R3d()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    root = 'F:\\CPR\\CPR_6\\single_view'
    path = os.path.join(root,'weight','R3d_3fc_finetune_30epoch_12_th_0.002_10_5')
    checkpoint  = torch.load(path)
    model.load_state_dict(checkpoint)
    inference(1)