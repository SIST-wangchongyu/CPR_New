import torch
import torchvision.models as models
import torch.nn as nn
# 加载R3D-18模型，并指定weights参数
# model = models.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, 5)
# print(model.fc.in_features)
# print(model)
class New_R3d(nn.Module):
    def __init__(self):
        super(New_R3d, self).__init__()
        #self.conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.r3d =  models.video.r3d_18(pretrained=True)
        # # 定义自己的全连接层

        fc1 = torch.nn.Linear(512, 256)
        fc2 = torch.nn.Linear(256, 128)
        fc3 = torch.nn.Linear(128, 5)

        self.r3d.fc = torch.nn.Sequential(fc1, torch.nn.ReLU(), fc2, torch.nn.ReLU(), fc3)
        # print(self.r3d)
        # self.r3d.fc = nn.Linear(in_features=512, out_features=5, bias=True)

    def forward(self, x):

        x = self.r3d(x)

        return x
