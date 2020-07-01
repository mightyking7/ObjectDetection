import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Darknet(nn.Module):
    """ YOLOv3 object detection model """
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)

