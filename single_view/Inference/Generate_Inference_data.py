
import torchvision.io as io
import numpy as np
import torch.nn.functional as F


def generate_data(path):
    video_path = path
    frame, _, _ = io.read_video(video_path)  # Frames: T,H,W,C
    print(frame.shape)
    frame= frame[:,100:700,480:1280,:]
    # np.save('new_data',frames)
    data = []
    label = []
    window_size =16
    stride =2


    num = frame.shape[0]
    print(num)

    for i in range(num):
        if num - i*stride <= 16:
            break
        frames =frame[i*stride:i*stride+16,:,:,:]
        print(frames.shape)
        frames = frames.permute(0, 3, 1, 2)
        frames = frames.float()
        frames = F.interpolate(frames,  size=(224, 224), mode='bilinear', align_corners=False)  # T,C,H,W
        frames_final = frames.permute(1,0,2,3)
        print(frames_final.shape)
        np.save(f"Inference_Data/video_{i}",frames_final)
if __name__ == '__main__':
    video_path = "CPR52B030人工呼吸.mp4"
    generate_data(video_path)
