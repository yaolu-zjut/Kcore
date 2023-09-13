# import os
# import random
# import numpy as np
# import torch
# import tqdm
# import torchvision.models as models
# from torch import nn
#
# from args import args
# import torch
# from Data.load_data import CIFAR10, CIFAR10_224


# def set_seed(seed):
#     print(f"Using seed {seed}")
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.deterministic = True
#
# set_seed(0)
# data = CIFAR10()
# data_224 = CIFAR10_224()
#
# # with torch.no_grad():
# #     for i, data in tqdm.tqdm(
# #             enumerate(data.train_loader), ascii=True, total=len(data.train_loader)
# #     ):
# #         images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
# #         print(target)
# #         break
#
# with torch.no_grad():
#     for i, data in tqdm.tqdm(
#             enumerate(data_224.train_loader), ascii=True, total=len(data_224.train_loader)
#     ):
#         images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
#         print(target)
#         break

# class BNFeatureHook():
#     def __init__(self, module):
#         self.hook = module.register_forward_hook(self.hook_fn)
#
#     def hook_fn(self, module, input, output):
#         print(input.shape)
#         nch = input[0].shape[1]
#         mean = input[0].mean([0, 2, 3])
#         var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
#         r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
#             module.running_mean.data - mean, 2)
#         self.r_feature = r_feature
#
#     def close(self):
#         self.hook.remove()
#
# model_teacher = models.__dict__['resnet18'](pretrained=True)
# print(model_teacher)
# targets_all = torch.LongTensor(np.arange(1000))
# print(targets_all)
# loss_r_feature_layers = []
# for module in model_teacher.modules():
#     if isinstance(module, nn.BatchNorm2d):
#         loss_r_feature_layers.append(BNFeatureHook(module))