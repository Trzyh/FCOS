
import torch
import tensorwatch as tw
from model.fcos import FCOSDetector

# 其实就两句话
model = FCOSDetector(mode="inference").cuda()
tw.draw_model(model, [1, 3, 640, 630])