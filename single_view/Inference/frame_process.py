import cv2
import numpy as np
root = 'F:\\CPR\\CPR_git\\single_view'
# 打开视频文件
video_capture = cv2.VideoCapture('CPR52B030人工呼吸.mp4')

# 获取视频的帧速率和宽高
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps)
print(width)
print(height)
# 创建一个VideoWriter对象，用于写入输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# 循环遍历每一帧
top_3 = np.load("top_3_action.npy", allow_pickle=True)
confidence = np.load("confidence_score.npy", allow_pickle=True)
print(len(top_3))
print(len(confidence))
counter =-1

while True:
    # 读取当前帧
    counter+=1
    index = int(counter/2)
    print(counter)
    ret, frame = video_capture.read()
    if index ==len(top_3)-1:
        break
    # 检查是否成功读取帧
    if not ret:
        break

    # 在左上角添加文本
    if top_3[index][0] == 0:
        text = "artificial respiration "
    elif top_3[index][0] == 1:
        text = "cardiac compression"
    elif top_3[index][0] == 2:
        text = "Check for pulse and respiration"
    elif top_3[index][0] == 3:
        text = "take off its clothes"
    elif top_3[index][0] == 4:
        text = "tap on the shoulders"
    top_1 = str(confidence[index][0])
    text = text+":"+top_1
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 50)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    if top_3[index][1] == 0:
        text = "artificial respiration "
    elif top_3[index][1] == 1:
        text = "cardiac compression"
    elif top_3[index][1] == 2:
        text = "Check for pulse and respiration"
    elif top_3[index][1] == 3:
        text = "take off its clothes"
    elif top_3[index][1] == 4:
        text = "tap on the shoulders"
    top_2 = str(confidence[index][1])
    text = text + ":" + top_2
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 150)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    if top_3[index][2] == 0:
        text = "artificial respiration "
    elif top_3[index][2] == 1:
        text = "cardiac compression"
    elif top_3[index][2] == 2:
        text = "Check for pulse and respiration"
    elif top_3[index][2] == 3:
        text = "take off its clothes"
    elif top_3[index][2] == 4:
        text = "tap on the shoulders"
    top_1 = str(confidence[index][2])
    text = text + ":" + top_1
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 250)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # 将当前帧写入输出视频
    output_video.write(frame)

    # 显示当前帧
    # cv2.imshow('Video', frame)
    # 等待用户按下 'q' 键退出程序    if cv2.waitKey(1) & 0xFF == ord('q'):
    #
    #         break


# 释放资源
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
