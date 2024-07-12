# @File: run_demo.py
# @Project: SceneTracker
# @Author : wangbo
# @Time : 2024.07.12

import numpy as np
import cv2
import torch
from model.model_scenetracker import SceneTracker
import run_test

def read_mp4(name_path):
    vidcap = cv2.VideoCapture(name_path)
    frames = []
    while (vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    return frames

print('SceneTracker demo start...')

model = SceneTracker()

pre_replace_list = [['module.', '']]
checkpoint = torch.load('exp/0-pretrain/scenetracker-odyssey-200k.pth')
for l in pre_replace_list:
    checkpoint = {k.replace(l[0], l[1]): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint, strict=True)
print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
model.eval().cuda()

run_test.validate_odyssey(model, split='demo')

print('Success!!!')

