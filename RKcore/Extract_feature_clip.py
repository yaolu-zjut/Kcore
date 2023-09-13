import clip
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import copy

from Data.load_data import CIFAR10
from args import args
import os
import random
import numpy as np

### Random seed
def set_seed(seed):
    print(f"Using seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

set_seed(0)
model, preprocess = clip.load("ViT-B/32", device=args.gpu)

''' organize the real dataset '''
total_feature = []
total_label = []
inter_feature = []
all_images = []
images_all = [[] for c in range(args.num_classes)]
feature_all = [[] for l in range(args.num_classes)]
labels_all = [[] for s in range(args.num_classes)]


# data = CIFAR10()
# count = 0
# with torch.no_grad():
#     for i, data in tqdm.tqdm(
#             enumerate(data.train_loader), ascii=True, total=len(data.train_loader)
#     ):
#         count += 1
#         images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
#         if count == 1:
#             print(target)
#         all_images.append(copy.deepcopy(images))
#         total_label.append(copy.deepcopy(target))
#
# for c in range(len(total_label)):
#     for num in range(total_label[c].shape[0]):
#         images_all[total_label[c][num]].append(all_images[c][num].unsqueeze(dim=0))
#
# total_imgs = []
# for c in range(args.num_classes):
#     imgs = torch.cat(images_all[c], dim=0).cuda(device=args.gpu)
#     total_imgs.append(imgs)
#
# save_path = 'experiment/' + args.arch + '/' + '%s' % (args.set) + '/'
# torch.save(total_imgs, save_path + 'all_imgs_{}_{}.pth'.format(args.arch, args.set))

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download the dataset
train_dataset = datasets.CIFAR10(root='/public/MountData/dataset/cifar10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='/public/MountData/dataset/cifar10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
)

count = 0
with torch.no_grad():
    for i, data in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        count += 1
        images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
        if count == 1:
            print(target)
        image_features = model.encode_image(images)  # torch.Size([500, 512])
        total_feature.append(copy.deepcopy(image_features))
        total_label.append(copy.deepcopy(target))

for c in range(len(total_label)):
    for num in range(total_label[c].shape[0]):
        labels_all[total_label[c][num]].append(total_label[c][num])
        feature_all[total_label[c][num]].append(total_feature[c][num].unsqueeze(dim=0))

total_feats = []
total_labs = []
total_imgs = []
for c in range(args.num_classes):
    feats = torch.cat(feature_all[c], dim=0).cuda(device=args.gpu)
    labs = torch.stack(labels_all[c])
    print('class c = %d: %d real images' % (c, feats.shape[0]))
    total_feats.append(feats)
    total_labs.append(labs)

# # save
# all_feats = {'feature': total_feats, 'label': total_labs, 'image': total_imgs}
all_feats = {'feature': total_feats, 'label': total_labs}
# feature: [tensor, tensor ..., tensor] 10 in total
# label: [tensor, tensor ..., tensor] 10 in total
save_path = 'experiment/' + args.arch + '/' + '%s' % (args.set) + '/'
torch.save(all_feats, save_path + 'all_feats_{}_{}.pth'.format(args.arch, args.set))
