# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch.nn as nn
import torch
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self):
        super(SwinUnet, self).__init__()
        self.swin_unet = SwinTransformerSys()

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    # def load_from(self):
    #     pretrained_path = "/home/tjc/pycharmprojects/swin-unet/model.pth"
    #     # if pretrained_path is not None:
    #     #     print("pretrained_path:{}".format(pretrained_path))
    #     #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     #     pretrained_dict = torch.load(pretrained_path, map_location=device)
    #     #     if "model" not in pretrained_dict:
    #     #         print("---start load pretrained modle by splitting---")
    #     #         pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
    #     #         for k in list(pretrained_dict.keys()):
    #     #             if "output" in k:
    #     #                 print("delete key:{}".format(k))
    #     #                 del pretrained_dict[k]
    #     #         msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
    #     #         # print(msg)
    #     #         return
    #     #     pretrained_dict = pretrained_dict['model']
    #     #     print("---start load pretrained modle of swin encoder---")
    #     #
    #     #     model_dict = self.swin_unet.state_dict()
    #     #     full_dict = copy.deepcopy(pretrained_dict)
    #     #     for k, v in pretrained_dict.items():
    #     #         if "layers." in k:
    #     #             current_layer_num = 3 - int(k[7:8])
    #     #             current_k = "layers_up." + str(current_layer_num) + k[8:]
    #     #             full_dict.update({current_k: v})
    #     #     for k in list(full_dict.keys()):
    #     #         if k in model_dict:
    #     #             if full_dict[k].shape != model_dict[k].shape:
    #     #                 print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
    #     #                 del full_dict[k]
    #     pretrained_dict = torch.load(pretrained_path, map_location='cuda')
    #     self.swin_unet.load_state_dict(pretrained_dict, strict=False)
    # #     # print(msg)
    # # else:
    #     print("none pretrain")

# if __name__ == '__main__':
#     model = SwinUnet().to('cpu')
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'Total Parameters: {total_params}')

# 参数量41380068
