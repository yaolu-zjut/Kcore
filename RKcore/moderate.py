import tqdm
import numpy as np
import argparse
import pickle
import torchvision.transforms as transforms
import os
from Data.load_data import CIFAR10
# from utils.resnet import ResNet50Extractor


def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))  # (50000, 2048, 1, 1) (50000,)
    prot = np.zeros((num_classes, features.shape[1]), dtype=features.dtype)  # (10, 2048)

    for i in range(num_classes):
        prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot


def get_distance(features, labels):  # features: (50000, 2048, 1, 1)
    prots = get_median(features, labels)  # (10, 2048)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))  # (50000, 2048)

    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels == i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features.squeeze() - prots_for_each_example, axis=1)

    return distance


def get_features(args):
    # obtain features of each sample
    import torchvision.models as models

    model = models.resnet50(pretrained=True)  # 注意这里有问题，他没有用cifar10的预训练
    import torch.nn as nn
    model = nn.Sequential(*list(model.children())[:-1])  # delete linear

    # model = ResNet50Extractor(num_classes=10)
    # model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(args.device)
    data_train = CIFAR10()


    targets, features = [], []
    for _, (img, target) in tqdm.tqdm(
            enumerate(data_train.train_loader), ascii=True, total=len(data_train.train_loader)
    ):
    # for _, (img, target) in tqdm(data_train.train_loader):
        targets.extend(target.numpy().tolist())

        img = img.to(args.device)
        feature = model(img).detach().cpu().numpy()
        features.extend([feature[i] for i in range(feature.shape[0])])

    features = np.array(features)
    targets = np.array(targets)
    # print(features.shape, targets.shape)
    return features, targets


def get_prune_idx(args, distance):
    low = 0.5 - args.rate / 2
    high = 0.5 + args.rate / 2

    sorted_idx = distance.argsort()
    low_idx = round(distance.shape[0] * low)
    high_idx = round(distance.shape[0] * high)

    ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))

    return ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50", help="backbone architecture")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--save", default="index", help="dir to save pruned image ids")
    parser.add_argument("--rate", type=float, default=0.2, help="selection ratio")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    features, targets = get_features(args)  # (50000, 2048, 1, 1) (50000,)
    distance = get_distance(features, targets)
    ids = get_prune_idx(args, distance)

    os.makedirs(args.save, exist_ok=True)
    save = os.path.join(args.save, f"{args.dataset}.bin")
    with open(save, "wb") as file:
        pickle.dump(ids, file)


if __name__ == "__main__":
    main()