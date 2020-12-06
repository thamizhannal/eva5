
import torch
import torch.nn as nn
from blocks import _make_resnet_backbone, _make_pretrained_resnext101_wsl
from blocks import _make_encoder, _make_resnext101_scratch, FeatureFusionBlock, Interpolate


class midas_encoder(nn.Module):
    def __init__(self, path="/content/gdrive/My Drive/SchoolOfAI_EVA/YoloV3_S13/YoloV3/weights/model-f46da743.pt"):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(midas_encoder, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained = _make_eyolov3-custom.cfgncoder( use_pretrained)
        '''
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )
        '''
        #if path:
            #self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        return layer_1, layer_2, layer_3, layer_4

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]
            print("####MODEL=".format(parameters))
        print("####parameters=".format(parameters))
        self.load_state_dict(parameters)

class midas_decoder(nn.Module):
    def __init__(self, features=256, non_negative=True, pre_load=False):
        super(midas_decoder, self).__init__()
        self.scratch = _make_resnext101_scratch([256, 512, 1024, 2048], features)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

    def forward(self, layer_1, layer_2, layer_3, layer_4):
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)

#################### yolo decoder ########################

def _FeatureConcat(layers):
        return torch.cat([i for i in layers], 1)


def _gt_darknet_child(x):
    if isinstance(list(Darknet.children())[0][x], nn.Sequential):
        return list(list(Darknet.children())[0][x].children())
    else:
        temp = []
        temp.append(list(Darknet.children())[0][x])
        return temp


def _gt_darknet_children(l):
    layer_list = []
    for x in l:
        layer_list.extend(_gt_darknet_child(x))
    return layer_list


def _get_seq_block(seqlist):
    module = nn.Sequential()
    for i, mod in enumerate(seqlist):
        if isinstance(mod, nn.Conv2d):
            module.add_module(module=mod, name='Conv2d' + '_' + str(i))
        elif isinstance(mod, nn.BatchNorm2d):
            module.add_module(module=mod, name='BatchNorm2d' + '_' + str(i))
        else:
            module.add_module(module=mod, name='others' + '_' + str(i))
        # print(i)
    return (module)


from model_yolo import *
from utils_yolo.utils import *
#cfg = '/content/gdrive/My Drive/SchoolOfAI_EVA/phase1_capstone/cfg_yolo/yolov3-custom.cfg'
cfg = 'C:\\Users\\tparamas\\PycharmProjects\\phase1_capstone\\cfg_yolo\\yolov3-custom.cfg'
Darknet = Darknet(cfg)

import torch.nn as nn
import numpy as np


class yolo_decoder(nn.Module):
    def __init__(self, pre_load=False, pre_load_pth=''):
        super(yolo_decoder, self).__init__()
        # anchors for 13x13, 26X26, 52X52
        if pre_load:
            checkpoint = torch.load(pre_load_pth)
            Darknet.load_state_dict(checkpoint['model'])
            print('loaded weight from', pre_load_pth)

        val = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
        anc_13 = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))[[6, 7, 8]]
        anc_26 = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))[[3, 4, 5]]
        anc_52 = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))[[0, 1, 2]]
        self.yolo_anchors = [anc_13, anc_26, anc_52]
        # 13x13 yolo layer
        self.yolo_13_bottle_neck = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False)
        self.yolo_13_path = _get_seq_block(_gt_darknet_children(list(range(84, 87))))
        self.yolo_13_tail = _get_seq_block(_gt_darknet_children(list(range(87, 89))))
        self.yolo_13 = YOLOLayer(anchors=anc_13,  # anchor list
                                 nc=4,  # number of classes
                                 img_size=(416, 416),  # (416, 416)
                                 yolo_index=0,  # 0, 1, 2...
                                 layers=[],  # output layers
                                 stride=32)
        # 26x26 yolo layer
        self.yolo_26_upsample = _get_seq_block(_gt_darknet_children(list(range(91, 93))))
        self.yolo_26_bottle_neck = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.yolo_26_path = _get_seq_block(_gt_darknet_children(list(range(94, 99))))
        self.yolo_26_tail = _get_seq_block(_gt_darknet_children(list(range(99, 101))))
        self.yolo_26 = YOLOLayer(anchors=anc_26,  # anchor list
                                 nc=4,  # number of classes
                                 img_size=(416, 416),  # (416, 416)
                                 yolo_index=0,  # 0, 1, 2...
                                 layers=[],  # output layers
                                 stride=16)

        # 52X52 yolo layer
        self.yolo_52_upsample = _get_seq_block(_gt_darknet_children(list(range(103, 105))))
        self.yolo_52_bottle_neck = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.yolo_52_path_tail = _get_seq_block(_gt_darknet_children(list(range(106, 113))))
        self.yolo_52 = YOLOLayer(anchors=anc_52,  # anchor list
                                 nc=4,  # number of classes
                                 img_size=(416, 416),  # (416, 416)
                                 yolo_index=0,  # 0, 1, 2...
                                 layers=[],  # output layers
                                 stride=8)

    def forward(self, EC2, EC3, out):
        # yolo 13
        out_bn = self.yolo_13_bottle_neck(out)
        out_13_path = self.yolo_13_path(out_bn)
        out_13_tail = self.yolo_13_tail(out_13_path)
        out_13_yolo = self.yolo_13(out_13_tail, [])
        # yolo 26
        out_26_upsample = self.yolo_26_upsample(out_13_path)
        out_EC3_bn = self.yolo_26_bottle_neck(EC3)
        out_26_FC = _FeatureConcat([out_26_upsample, out_EC3_bn])
        out_26_path = self.yolo_26_path(out_26_FC)
        out_26_tail = self.yolo_26_tail(out_26_path)
        out_26_yolo = self.yolo_26(out_26_tail, [])
        # yolo 52
        out_52_upsample = self.yolo_52_upsample(out_26_path)
        out_EC2_bn = self.yolo_52_bottle_neck(EC2)
        out_52_FC = _FeatureConcat([out_52_upsample, out_EC2_bn])
        out_52_tail = self.yolo_52_path_tail(out_52_FC)
        out_52_yolo = self.yolo_52(out_52_tail, [])
        return out_13_yolo, out_26_yolo, out_52_yolo

# final fork model
# D:/ML/EVA/JEDi/Midas/model-f46da743.pt - depth decoder
# D:/ML/EVA/JEDI/YoloV3master/last_ppe.pt - yolo decoder
class fork(nn.Module):
    def __init__(self, depth_freeze=False, yolo_freeze=False, depth_preload_pth='', yolo_preload_pth=''):
        super(fork, self).__init__()
        # anchors for yolo loss calculation
        val = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
        anc_13 = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))[[6, 7, 8]]
        anc_26 = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))[[3, 4, 5]]
        anc_52 = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))[[0, 1, 2]]
        self.yolo_anchors = [anc_13, anc_26, anc_52]
        # encoder
        self.encoder = midas_encoder()
        # depth decoder with preload
        self.decoder = midas_decoder(features=256)
        if depth_freeze:
            child_counter = 0
            for child in self.decoder.children():
                print("depth_layer", child_counter, "was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                child_counter += 1

        if depth_preload_pth != '':
            checkpoint = torch.load(depth_preload_pth)
            dl_list = ["pretrained.layer1.0.weight", "pretrained.layer1.1.weight", "pretrained.layer1.1.bias", "pretrained.layer1.1.running_mean", "pretrained.layer1.1.running_var", "pretrained.layer1.1.num_batches_tracked", "pretrained.layer1.4.0.conv1.weight", "pretrained.layer1.4.0.bn1.weight", "pretrained.layer1.4.0.bn1.bias", "pretrained.layer1.4.0.bn1.running_mean", "pretrained.layer1.4.0.bn1.running_var", "pretrained.layer1.4.0.bn1.num_batches_tracked", "pretrained.layer1.4.0.conv2.weight", "pretrained.layer1.4.0.bn2.weight", "pretrained.layer1.4.0.bn2.bias", "pretrained.layer1.4.0.bn2.running_mean", "pretrained.layer1.4.0.bn2.running_var", "pretrained.layer1.4.0.bn2.num_batches_tracked", "pretrained.layer1.4.0.conv3.weight", "pretrained.layer1.4.0.bn3.weight", "pretrained.layer1.4.0.bn3.bias", "pretrained.layer1.4.0.bn3.running_mean", "pretrained.layer1.4.0.bn3.running_var", "pretrained.layer1.4.0.bn3.num_batches_tracked", "pretrained.layer1.4.0.downsample.0.weight", "pretrained.layer1.4.0.downsample.1.weight", "pretrained.layer1.4.0.downsample.1.bias", "pretrained.layer1.4.0.downsample.1.running_mean", "pretrained.layer1.4.0.downsample.1.running_var", "pretrained.layer1.4.0.downsample.1.num_batches_tracked", "pretrained.layer1.4.1.conv1.weight", "pretrained.layer1.4.1.bn1.weight", "pretrained.layer1.4.1.bn1.bias", "pretrained.layer1.4.1.bn1.running_mean", "pretrained.layer1.4.1.bn1.running_var", "pretrained.layer1.4.1.bn1.num_batches_tracked", "pretrained.layer1.4.1.conv2.weight", "pretrained.layer1.4.1.bn2.weight", "pretrained.layer1.4.1.bn2.bias", "pretrained.layer1.4.1.bn2.running_mean", "pretrained.layer1.4.1.bn2.running_var", "pretrained.layer1.4.1.bn2.num_batches_tracked", "pretrained.layer1.4.1.conv3.weight", "pretrained.layer1.4.1.bn3.weight", "pretrained.layer1.4.1.bn3.bias", "pretrained.layer1.4.1.bn3.running_mean", "pretrained.layer1.4.1.bn3.running_var", "pretrained.layer1.4.1.bn3.num_batches_tracked", "pretrained.layer1.4.2.conv1.weight", "pretrained.layer1.4.2.bn1.weight", "pretrained.layer1.4.2.bn1.bias", "pretrained.layer1.4.2.bn1.running_mean", "pretrained.layer1.4.2.bn1.running_var", "pretrained.layer1.4.2.bn1.num_batches_tracked", "pretrained.layer1.4.2.conv2.weight", "pretrained.layer1.4.2.bn2.weight", "pretrained.layer1.4.2.bn2.bias", "pretrained.layer1.4.2.bn2.running_mean", "pretrained.layer1.4.2.bn2.running_var", "pretrained.layer1.4.2.bn2.num_batches_tracked", "pretrained.layer1.4.2.conv3.weight", "pretrained.layer1.4.2.bn3.weight", "pretrained.layer1.4.2.bn3.bias", "pretrained.layer1.4.2.bn3.running_mean", "pretrained.layer1.4.2.bn3.running_var", "pretrained.layer1.4.2.bn3.num_batches_tracked", "pretrained.layer2.0.conv1.weight", "pretrained.layer2.0.bn1.weight", "pretrained.layer2.0.bn1.bias", "pretrained.layer2.0.bn1.running_mean", "pretrained.layer2.0.bn1.running_var", "pretrained.layer2.0.bn1.num_batches_tracked", "pretrained.layer2.0.conv2.weight", "pretrained.layer2.0.bn2.weight", "pretrained.layer2.0.bn2.bias", "pretrained.layer2.0.bn2.running_mean", "pretrained.layer2.0.bn2.running_var", "pretrained.layer2.0.bn2.num_batches_tracked", "pretrained.layer2.0.conv3.weight", "pretrained.layer2.0.bn3.weight", "pretrained.layer2.0.bn3.bias", "pretrained.layer2.0.bn3.running_mean", "pretrained.layer2.0.bn3.running_var", "pretrained.layer2.0.bn3.num_batches_tracked", "pretrained.layer2.0.downsample.0.weight", "pretrained.layer2.0.downsample.1.weight", "pretrained.layer2.0.downsample.1.bias", "pretrained.layer2.0.downsample.1.running_mean", "pretrained.layer2.0.downsample.1.running_var", "pretrained.layer2.0.downsample.1.num_batches_tracked", "pretrained.layer2.1.conv1.weight", "pretrained.layer2.1.bn1.weight", "pretrained.layer2.1.bn1.bias", "pretrained.layer2.1.bn1.running_mean", "pretrained.layer2.1.bn1.running_var", "pretrained.layer2.1.bn1.num_batches_tracked", "pretrained.layer2.1.conv2.weight", "pretrained.layer2.1.bn2.weight", "pretrained.layer2.1.bn2.bias", "pretrained.layer2.1.bn2.running_mean", "pretrained.layer2.1.bn2.running_var", "pretrained.layer2.1.bn2.num_batches_tracked", "pretrained.layer2.1.conv3.weight", "pretrained.layer2.1.bn3.weight", "pretrained.layer2.1.bn3.bias", "pretrained.layer2.1.bn3.running_mean", "pretrained.layer2.1.bn3.running_var", "pretrained.layer2.1.bn3.num_batches_tracked", "pretrained.layer2.2.conv1.weight", "pretrained.layer2.2.bn1.weight", "pretrained.layer2.2.bn1.bias", "pretrained.layer2.2.bn1.running_mean", "pretrained.layer2.2.bn1.running_var", "pretrained.layer2.2.bn1.num_batches_tracked", "pretrained.layer2.2.conv2.weight", "pretrained.layer2.2.bn2.weight", "pretrained.layer2.2.bn2.bias", "pretrained.layer2.2.bn2.running_mean", "pretrained.layer2.2.bn2.running_var", "pretrained.layer2.2.bn2.num_batches_tracked", "pretrained.layer2.2.conv3.weight", "pretrained.layer2.2.bn3.weight", "pretrained.layer2.2.bn3.bias", "pretrained.layer2.2.bn3.running_mean", "pretrained.layer2.2.bn3.running_var", "pretrained.layer2.2.bn3.num_batches_tracked", "pretrained.layer2.3.conv1.weight", "pretrained.layer2.3.bn1.weight", "pretrained.layer2.3.bn1.bias", "pretrained.layer2.3.bn1.running_mean", "pretrained.layer2.3.bn1.running_var", "pretrained.layer2.3.bn1.num_batches_tracked", "pretrained.layer2.3.conv2.weight", "pretrained.layer2.3.bn2.weight", "pretrained.layer2.3.bn2.bias", "pretrained.layer2.3.bn2.running_mean", "pretrained.layer2.3.bn2.running_var", "pretrained.layer2.3.bn2.num_batches_tracked", "pretrained.layer2.3.conv3.weight", "pretrained.layer2.3.bn3.weight", "pretrained.layer2.3.bn3.bias", "pretrained.layer2.3.bn3.running_mean", "pretrained.layer2.3.bn3.running_var", "pretrained.layer2.3.bn3.num_batches_tracked", "pretrained.layer3.0.conv1.weight", "pretrained.layer3.0.bn1.weight", "pretrained.layer3.0.bn1.bias", "pretrained.layer3.0.bn1.running_mean", "pretrained.layer3.0.bn1.running_var", "pretrained.layer3.0.bn1.num_batches_tracked", "pretrained.layer3.0.conv2.weight", "pretrained.layer3.0.bn2.weight", "pretrained.layer3.0.bn2.bias", "pretrained.layer3.0.bn2.running_mean", "pretrained.layer3.0.bn2.running_var", "pretrained.layer3.0.bn2.num_batches_tracked", "pretrained.layer3.0.conv3.weight", "pretrained.layer3.0.bn3.weight", "pretrained.layer3.0.bn3.bias", "pretrained.layer3.0.bn3.running_mean", "pretrained.layer3.0.bn3.running_var", "pretrained.layer3.0.bn3.num_batches_tracked", "pretrained.layer3.0.downsample.0.weight", "pretrained.layer3.0.downsample.1.weight", "pretrained.layer3.0.downsample.1.bias", "pretrained.layer3.0.downsample.1.running_mean", "pretrained.layer3.0.downsample.1.running_var", "pretrained.layer3.0.downsample.1.num_batches_tracked", "pretrained.layer3.1.conv1.weight", "pretrained.layer3.1.bn1.weight", "pretrained.layer3.1.bn1.bias", "pretrained.layer3.1.bn1.running_mean", "pretrained.layer3.1.bn1.running_var", "pretrained.layer3.1.bn1.num_batches_tracked", "pretrained.layer3.1.conv2.weight", "pretrained.layer3.1.bn2.weight", "pretrained.layer3.1.bn2.bias", "pretrained.layer3.1.bn2.running_mean", "pretrained.layer3.1.bn2.running_var", "pretrained.layer3.1.bn2.num_batches_tracked", "pretrained.layer3.1.conv3.weight", "pretrained.layer3.1.bn3.weight", "pretrained.layer3.1.bn3.bias", "pretrained.layer3.1.bn3.running_mean", "pretrained.layer3.1.bn3.running_var", "pretrained.layer3.1.bn3.num_batches_tracked", "pretrained.layer3.2.conv1.weight", "pretrained.layer3.2.bn1.weight", "pretrained.layer3.2.bn1.bias", "pretrained.layer3.2.bn1.running_mean", "pretrained.layer3.2.bn1.running_var", "pretrained.layer3.2.bn1.num_batches_tracked", "pretrained.layer3.2.conv2.weight", "pretrained.layer3.2.bn2.weight", "pretrained.layer3.2.bn2.bias", "pretrained.layer3.2.bn2.running_mean", "pretrained.layer3.2.bn2.running_var", "pretrained.layer3.2.bn2.num_batches_tracked", "pretrained.layer3.2.conv3.weight", "pretrained.layer3.2.bn3.weight", "pretrained.layer3.2.bn3.bias", "pretrained.layer3.2.bn3.running_mean", "pretrained.layer3.2.bn3.running_var", "pretrained.layer3.2.bn3.num_batches_tracked", "pretrained.layer3.3.conv1.weight", "pretrained.layer3.3.bn1.weight", "pretrained.layer3.3.bn1.bias", "pretrained.layer3.3.bn1.running_mean", "pretrained.layer3.3.bn1.running_var", "pretrained.layer3.3.bn1.num_batches_tracked", "pretrained.layer3.3.conv2.weight", "pretrained.layer3.3.bn2.weight", "pretrained.layer3.3.bn2.bias", "pretrained.layer3.3.bn2.running_mean", "pretrained.layer3.3.bn2.running_var", "pretrained.layer3.3.bn2.num_batches_tracked", "pretrained.layer3.3.conv3.weight", "pretrained.layer3.3.bn3.weight", "pretrained.layer3.3.bn3.bias", "pretrained.layer3.3.bn3.running_mean", "pretrained.layer3.3.bn3.running_var", "pretrained.layer3.3.bn3.num_batches_tracked", "pretrained.layer3.4.conv1.weight", "pretrained.layer3.4.bn1.weight", "pretrained.layer3.4.bn1.bias", "pretrained.layer3.4.bn1.running_mean", "pretrained.layer3.4.bn1.running_var", "pretrained.layer3.4.bn1.num_batches_tracked", "pretrained.layer3.4.conv2.weight", "pretrained.layer3.4.bn2.weight", "pretrained.layer3.4.bn2.bias", "pretrained.layer3.4.bn2.running_mean", "pretrained.layer3.4.bn2.running_var", "pretrained.layer3.4.bn2.num_batches_tracked", "pretrained.layer3.4.conv3.weight", "pretrained.layer3.4.bn3.weight", "pretrained.layer3.4.bn3.bias", "pretrained.layer3.4.bn3.running_mean", "pretrained.layer3.4.bn3.running_var", "pretrained.layer3.4.bn3.num_batches_tracked", "pretrained.layer3.5.conv1.weight", "pretrained.layer3.5.bn1.weight", "pretrained.layer3.5.bn1.bias", "pretrained.layer3.5.bn1.running_mean", "pretrained.layer3.5.bn1.running_var", "pretrained.layer3.5.bn1.num_batches_tracked", "pretrained.layer3.5.conv2.weight", "pretrained.layer3.5.bn2.weight", "pretrained.layer3.5.bn2.bias", "pretrained.layer3.5.bn2.running_mean", "pretrained.layer3.5.bn2.running_var", "pretrained.layer3.5.bn2.num_batches_tracked", "pretrained.layer3.5.conv3.weight", "pretrained.layer3.5.bn3.weight", "pretrained.layer3.5.bn3.bias", "pretrained.layer3.5.bn3.running_mean", "pretrained.layer3.5.bn3.running_var", "pretrained.layer3.5.bn3.num_batches_tracked", "pretrained.layer3.6.conv1.weight", "pretrained.layer3.6.bn1.weight", "pretrained.layer3.6.bn1.bias", "pretrained.layer3.6.bn1.running_mean", "pretrained.layer3.6.bn1.running_var", "pretrained.layer3.6.bn1.num_batches_tracked", "pretrained.layer3.6.conv2.weight", "pretrained.layer3.6.bn2.weight", "pretrained.layer3.6.bn2.bias", "pretrained.layer3.6.bn2.running_mean", "pretrained.layer3.6.bn2.running_var", "pretrained.layer3.6.bn2.num_batches_tracked", "pretrained.layer3.6.conv3.weight", "pretrained.layer3.6.bn3.weight", "pretrained.layer3.6.bn3.bias", "pretrained.layer3.6.bn3.running_mean", "pretrained.layer3.6.bn3.running_var", "pretrained.layer3.6.bn3.num_batches_tracked", "pretrained.layer3.7.conv1.weight", "pretrained.layer3.7.bn1.weight", "pretrained.layer3.7.bn1.bias", "pretrained.layer3.7.bn1.running_mean", "pretrained.layer3.7.bn1.running_var", "pretrained.layer3.7.bn1.num_batches_tracked", "pretrained.layer3.7.conv2.weight", "pretrained.layer3.7.bn2.weight", "pretrained.layer3.7.bn2.bias", "pretrained.layer3.7.bn2.running_mean", "pretrained.layer3.7.bn2.running_var", "pretrained.layer3.7.bn2.num_batches_tracked", "pretrained.layer3.7.conv3.weight", "pretrained.layer3.7.bn3.weight", "pretrained.layer3.7.bn3.bias", "pretrained.layer3.7.bn3.running_mean", "pretrained.layer3.7.bn3.running_var", "pretrained.layer3.7.bn3.num_batches_tracked", "pretrained.layer3.8.conv1.weight", "pretrained.layer3.8.bn1.weight", "pretrained.layer3.8.bn1.bias", "pretrained.layer3.8.bn1.running_mean", "pretrained.layer3.8.bn1.running_var", "pretrained.layer3.8.bn1.num_batches_tracked", "pretrained.layer3.8.conv2.weight", "pretrained.layer3.8.bn2.weight", "pretrained.layer3.8.bn2.bias", "pretrained.layer3.8.bn2.running_mean", "pretrained.layer3.8.bn2.running_var", "pretrained.layer3.8.bn2.num_batches_tracked", "pretrained.layer3.8.conv3.weight", "pretrained.layer3.8.bn3.weight", "pretrained.layer3.8.bn3.bias", "pretrained.layer3.8.bn3.running_mean", "pretrained.layer3.8.bn3.running_var", "pretrained.layer3.8.bn3.num_batches_tracked", "pretrained.layer3.9.conv1.weight", "pretrained.layer3.9.bn1.weight", "pretrained.layer3.9.bn1.bias", "pretrained.layer3.9.bn1.running_mean", "pretrained.layer3.9.bn1.running_var", "pretrained.layer3.9.bn1.num_batches_tracked", "pretrained.layer3.9.conv2.weight", "pretrained.layer3.9.bn2.weight", "pretrained.layer3.9.bn2.bias", "pretrained.layer3.9.bn2.running_mean", "pretrained.layer3.9.bn2.running_var", "pretrained.layer3.9.bn2.num_batches_tracked", "pretrained.layer3.9.conv3.weight", "pretrained.layer3.9.bn3.weight", "pretrained.layer3.9.bn3.bias", "pretrained.layer3.9.bn3.running_mean", "pretrained.layer3.9.bn3.running_var", "pretrained.layer3.9.bn3.num_batches_tracked", "pretrained.layer3.10.conv1.weight", "pretrained.layer3.10.bn1.weight", "pretrained.layer3.10.bn1.bias", "pretrained.layer3.10.bn1.running_mean", "pretrained.layer3.10.bn1.running_var", "pretrained.layer3.10.bn1.num_batches_tracked", "pretrained.layer3.10.conv2.weight", "pretrained.layer3.10.bn2.weight", "pretrained.layer3.10.bn2.bias", "pretrained.layer3.10.bn2.running_mean", "pretrained.layer3.10.bn2.running_var", "pretrained.layer3.10.bn2.num_batches_tracked", "pretrained.layer3.10.conv3.weight", "pretrained.layer3.10.bn3.weight", "pretrained.layer3.10.bn3.bias", "pretrained.layer3.10.bn3.running_mean", "pretrained.layer3.10.bn3.running_var", "pretrained.layer3.10.bn3.num_batches_tracked", "pretrained.layer3.11.conv1.weight", "pretrained.layer3.11.bn1.weight", "pretrained.layer3.11.bn1.bias", "pretrained.layer3.11.bn1.running_mean", "pretrained.layer3.11.bn1.running_var", "pretrained.layer3.11.bn1.num_batches_tracked", "pretrained.layer3.11.conv2.weight", "pretrained.layer3.11.bn2.weight", "pretrained.layer3.11.bn2.bias", "pretrained.layer3.11.bn2.running_mean", "pretrained.layer3.11.bn2.running_var", "pretrained.layer3.11.bn2.num_batches_tracked", "pretrained.layer3.11.conv3.weight", "pretrained.layer3.11.bn3.weight", "pretrained.layer3.11.bn3.bias", "pretrained.layer3.11.bn3.running_mean", "pretrained.layer3.11.bn3.running_var", "pretrained.layer3.11.bn3.num_batches_tracked", "pretrained.layer3.12.conv1.weight", "pretrained.layer3.12.bn1.weight", "pretrained.layer3.12.bn1.bias", "pretrained.layer3.12.bn1.running_mean", "pretrained.layer3.12.bn1.running_var", "pretrained.layer3.12.bn1.num_batches_tracked", "pretrained.layer3.12.conv2.weight", "pretrained.layer3.12.bn2.weight", "pretrained.layer3.12.bn2.bias", "pretrained.layer3.12.bn2.running_mean", "pretrained.layer3.12.bn2.running_var", "pretrained.layer3.12.bn2.num_batches_tracked", "pretrained.layer3.12.conv3.weight", "pretrained.layer3.12.bn3.weight", "pretrained.layer3.12.bn3.bias", "pretrained.layer3.12.bn3.running_mean", "pretrained.layer3.12.bn3.running_var", "pretrained.layer3.12.bn3.num_batches_tracked", "pretrained.layer3.13.conv1.weight", "pretrained.layer3.13.bn1.weight", "pretrained.layer3.13.bn1.bias", "pretrained.layer3.13.bn1.running_mean", "pretrained.layer3.13.bn1.running_var", "pretrained.layer3.13.bn1.num_batches_tracked", "pretrained.layer3.13.conv2.weight", "pretrained.layer3.13.bn2.weight", "pretrained.layer3.13.bn2.bias", "pretrained.layer3.13.bn2.running_mean", "pretrained.layer3.13.bn2.running_var", "pretrained.layer3.13.bn2.num_batches_tracked", "pretrained.layer3.13.conv3.weight", "pretrained.layer3.13.bn3.weight", "pretrained.layer3.13.bn3.bias", "pretrained.layer3.13.bn3.running_mean", "pretrained.layer3.13.bn3.running_var", "pretrained.layer3.13.bn3.num_batches_tracked", "pretrained.layer3.14.conv1.weight", "pretrained.layer3.14.bn1.weight", "pretrained.layer3.14.bn1.bias", "pretrained.layer3.14.bn1.running_mean", "pretrained.layer3.14.bn1.running_var", "pretrained.layer3.14.bn1.num_batches_tracked", "pretrained.layer3.14.conv2.weight", "pretrained.layer3.14.bn2.weight", "pretrained.layer3.14.bn2.bias", "pretrained.layer3.14.bn2.running_mean", "pretrained.layer3.14.bn2.running_var", "pretrained.layer3.14.bn2.num_batches_tracked", "pretrained.layer3.14.conv3.weight", "pretrained.layer3.14.bn3.weight", "pretrained.layer3.14.bn3.bias", "pretrained.layer3.14.bn3.running_mean", "pretrained.layer3.14.bn3.running_var", "pretrained.layer3.14.bn3.num_batches_tracked", "pretrained.layer3.15.conv1.weight", "pretrained.layer3.15.bn1.weight", "pretrained.layer3.15.bn1.bias", "pretrained.layer3.15.bn1.running_mean", "pretrained.layer3.15.bn1.running_var", "pretrained.layer3.15.bn1.num_batches_tracked", "pretrained.layer3.15.conv2.weight", "pretrained.layer3.15.bn2.weight", "pretrained.layer3.15.bn2.bias", "pretrained.layer3.15.bn2.running_mean", "pretrained.layer3.15.bn2.running_var", "pretrained.layer3.15.bn2.num_batches_tracked", "pretrained.layer3.15.conv3.weight", "pretrained.layer3.15.bn3.weight", "pretrained.layer3.15.bn3.bias", "pretrained.layer3.15.bn3.running_mean", "pretrained.layer3.15.bn3.running_var", "pretrained.layer3.15.bn3.num_batches_tracked", "pretrained.layer3.16.conv1.weight", "pretrained.layer3.16.bn1.weight", "pretrained.layer3.16.bn1.bias", "pretrained.layer3.16.bn1.running_mean", "pretrained.layer3.16.bn1.running_var", "pretrained.layer3.16.bn1.num_batches_tracked", "pretrained.layer3.16.conv2.weight", "pretrained.layer3.16.bn2.weight", "pretrained.layer3.16.bn2.bias", "pretrained.layer3.16.bn2.running_mean", "pretrained.layer3.16.bn2.running_var", "pretrained.layer3.16.bn2.num_batches_tracked", "pretrained.layer3.16.conv3.weight", "pretrained.layer3.16.bn3.weight", "pretrained.layer3.16.bn3.bias", "pretrained.layer3.16.bn3.running_mean", "pretrained.layer3.16.bn3.running_var", "pretrained.layer3.16.bn3.num_batches_tracked", "pretrained.layer3.17.conv1.weight", "pretrained.layer3.17.bn1.weight", "pretrained.layer3.17.bn1.bias", "pretrained.layer3.17.bn1.running_mean", "pretrained.layer3.17.bn1.running_var", "pretrained.layer3.17.bn1.num_batches_tracked", "pretrained.layer3.17.conv2.weight", "pretrained.layer3.17.bn2.weight", "pretrained.layer3.17.bn2.bias", "pretrained.layer3.17.bn2.running_mean", "pretrained.layer3.17.bn2.running_var", "pretrained.layer3.17.bn2.num_batches_tracked", "pretrained.layer3.17.conv3.weight", "pretrained.layer3.17.bn3.weight", "pretrained.layer3.17.bn3.bias", "pretrained.layer3.17.bn3.running_mean", "pretrained.layer3.17.bn3.running_var", "pretrained.layer3.17.bn3.num_batches_tracked", "pretrained.layer3.18.conv1.weight", "pretrained.layer3.18.bn1.weight", "pretrained.layer3.18.bn1.bias", "pretrained.layer3.18.bn1.running_mean", "pretrained.layer3.18.bn1.running_var", "pretrained.layer3.18.bn1.num_batches_tracked", "pretrained.layer3.18.conv2.weight", "pretrained.layer3.18.bn2.weight", "pretrained.layer3.18.bn2.bias", "pretrained.layer3.18.bn2.running_mean", "pretrained.layer3.18.bn2.running_var", "pretrained.layer3.18.bn2.num_batches_tracked", "pretrained.layer3.18.conv3.weight", "pretrained.layer3.18.bn3.weight", "pretrained.layer3.18.bn3.bias", "pretrained.layer3.18.bn3.running_mean", "pretrained.layer3.18.bn3.running_var", "pretrained.layer3.18.bn3.num_batches_tracked", "pretrained.layer3.19.conv1.weight", "pretrained.layer3.19.bn1.weight", "pretrained.layer3.19.bn1.bias", "pretrained.layer3.19.bn1.running_mean", "pretrained.layer3.19.bn1.running_var", "pretrained.layer3.19.bn1.num_batches_tracked", "pretrained.layer3.19.conv2.weight", "pretrained.layer3.19.bn2.weight", "pretrained.layer3.19.bn2.bias", "pretrained.layer3.19.bn2.running_mean", "pretrained.layer3.19.bn2.running_var", "pretrained.layer3.19.bn2.num_batches_tracked", "pretrained.layer3.19.conv3.weight", "pretrained.layer3.19.bn3.weight", "pretrained.layer3.19.bn3.bias", "pretrained.layer3.19.bn3.running_mean", "pretrained.layer3.19.bn3.running_var", "pretrained.layer3.19.bn3.num_batches_tracked", "pretrained.layer3.20.conv1.weight", "pretrained.layer3.20.bn1.weight", "pretrained.layer3.20.bn1.bias", "pretrained.layer3.20.bn1.running_mean", "pretrained.layer3.20.bn1.running_var", "pretrained.layer3.20.bn1.num_batches_tracked", "pretrained.layer3.20.conv2.weight", "pretrained.layer3.20.bn2.weight", "pretrained.layer3.20.bn2.bias", "pretrained.layer3.20.bn2.running_mean", "pretrained.layer3.20.bn2.running_var", "pretrained.layer3.20.bn2.num_batches_tracked", "pretrained.layer3.20.conv3.weight", "pretrained.layer3.20.bn3.weight", "pretrained.layer3.20.bn3.bias", "pretrained.layer3.20.bn3.running_mean", "pretrained.layer3.20.bn3.running_var", "pretrained.layer3.20.bn3.num_batches_tracked", "pretrained.layer3.21.conv1.weight", "pretrained.layer3.21.bn1.weight", "pretrained.layer3.21.bn1.bias", "pretrained.layer3.21.bn1.running_mean", "pretrained.layer3.21.bn1.running_var", "pretrained.layer3.21.bn1.num_batches_tracked", "pretrained.layer3.21.conv2.weight", "pretrained.layer3.21.bn2.weight", "pretrained.layer3.21.bn2.bias", "pretrained.layer3.21.bn2.running_mean", "pretrained.layer3.21.bn2.running_var", "pretrained.layer3.21.bn2.num_batches_tracked", "pretrained.layer3.21.conv3.weight", "pretrained.layer3.21.bn3.weight", "pretrained.layer3.21.bn3.bias", "pretrained.layer3.21.bn3.running_mean", "pretrained.layer3.21.bn3.running_var", "pretrained.layer3.21.bn3.num_batches_tracked", "pretrained.layer3.22.conv1.weight", "pretrained.layer3.22.bn1.weight", "pretrained.layer3.22.bn1.bias", "pretrained.layer3.22.bn1.running_mean", "pretrained.layer3.22.bn1.running_var", "pretrained.layer3.22.bn1.num_batches_tracked", "pretrained.layer3.22.conv2.weight", "pretrained.layer3.22.bn2.weight", "pretrained.layer3.22.bn2.bias", "pretrained.layer3.22.bn2.running_mean", "pretrained.layer3.22.bn2.running_var", "pretrained.layer3.22.bn2.num_batches_tracked", "pretrained.layer3.22.conv3.weight", "pretrained.layer3.22.bn3.weight", "pretrained.layer3.22.bn3.bias", "pretrained.layer3.22.bn3.running_mean", "pretrained.layer3.22.bn3.running_var", "pretrained.layer3.22.bn3.num_batches_tracked", "pretrained.layer4.0.conv1.weight", "pretrained.layer4.0.bn1.weight", "pretrained.layer4.0.bn1.bias", "pretrained.layer4.0.bn1.running_mean", "pretrained.layer4.0.bn1.running_var", "pretrained.layer4.0.bn1.num_batches_tracked", "pretrained.layer4.0.conv2.weight", "pretrained.layer4.0.bn2.weight", "pretrained.layer4.0.bn2.bias", "pretrained.layer4.0.bn2.running_mean", "pretrained.layer4.0.bn2.running_var", "pretrained.layer4.0.bn2.num_batches_tracked", "pretrained.layer4.0.conv3.weight", "pretrained.layer4.0.bn3.weight", "pretrained.layer4.0.bn3.bias", "pretrained.layer4.0.bn3.running_mean", "pretrained.layer4.0.bn3.running_var", "pretrained.layer4.0.bn3.num_batches_tracked", "pretrained.layer4.0.downsample.0.weight", "pretrained.layer4.0.downsample.1.weight", "pretrained.layer4.0.downsample.1.bias", "pretrained.layer4.0.downsample.1.running_mean", "pretrained.layer4.0.downsample.1.running_var", "pretrained.layer4.0.downsample.1.num_batches_tracked", "pretrained.layer4.1.conv1.weight", "pretrained.layer4.1.bn1.weight", "pretrained.layer4.1.bn1.bias", "pretrained.layer4.1.bn1.running_mean", "pretrained.layer4.1.bn1.running_var", "pretrained.layer4.1.bn1.num_batches_tracked", "pretrained.layer4.1.conv2.weight", "pretrained.layer4.1.bn2.weight", "pretrained.layer4.1.bn2.bias", "pretrained.layer4.1.bn2.running_mean", "pretrained.layer4.1.bn2.running_var", "pretrained.layer4.1.bn2.num_batches_tracked", "pretrained.layer4.1.conv3.weight", "pretrained.layer4.1.bn3.weight", "pretrained.layer4.1.bn3.bias", "pretrained.layer4.1.bn3.running_mean", "pretrained.layer4.1.bn3.running_var", "pretrained.layer4.1.bn3.num_batches_tracked", "pretrained.layer4.2.conv1.weight", "pretrained.layer4.2.bn1.weight", "pretrained.layer4.2.bn1.bias", "pretrained.layer4.2.bn1.running_mean", "pretrained.layer4.2.bn1.running_var", "pretrained.layer4.2.bn1.num_batches_tracked", "pretrained.layer4.2.conv2.weight", "pretrained.layer4.2.bn2.weight", "pretrained.layer4.2.bn2.bias", "pretrained.layer4.2.bn2.running_mean", "pretrained.layer4.2.bn2.running_var", "pretrained.layer4.2.bn2.num_batches_tracked", "pretrained.layer4.2.conv3.weight", "pretrained.layer4.2.bn3.weight", "pretrained.layer4.2.bn3.bias", "pretrained.layer4.2.bn3.running_mean", "pretrained.layer4.2.bn3.running_var", "pretrained.layer4.2.bn3.num_batches_tracked"]
            for x in dl_list:
                del checkpoint[x]
            self.decoder.load_state_dict(checkpoint)
            print('depth_decoder loaded from', depth_preload_pth)

        if yolo_preload_pth != '':
            self.yolo_decoder = yolo_decoder(pre_load=True, pre_load_pth=yolo_preload_pth)
            print('yolo_decoder loaded from', yolo_preload_pth)
        else:
            self.yolo_decoder = yolo_decoder(pre_load=False, pre_load_pth='')
        if yolo_freeze:
            child_counter = 0
            for child in self.yolo_decoder.children():
                print("yolo_layer", child_counter, "was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                child_counter += 1

    def forward(self, x):
        EC1, EC2, EC3, out = self.encoder(x)
        depth_out = self.decoder(EC1, EC2, EC3, out)
        out_13_yolo, out_26_yolo, out_52_yolo = self.yolo_decoder(EC2, EC3, out)
        return depth_out, out_13_yolo, out_26_yolo, out_52_yolo