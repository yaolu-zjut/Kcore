import os
import pickle
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from utils.resnet_imagenet import *
from Data.load_data import CIFAR10,ImageNet,CIFAR100
from Data.moderateds_dataset import CIFAR10Core
from args import args
import datetime
from trainer.amp_trainer_dali import train_ImageNet, validate_ImageNet
from trainer.trainer import validate, train
from utils.resnet import ResNet18
from utils.resnet_cifar import resnet20
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from utils.warmup_lr import cosine_lr
import random
from torch.utils.data import Subset, ConcatDataset


def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch + '/' + args.set, exist_ok=True)
    logger = get_logger('pretrained_model/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    num = 999
    selected_dataset = torch.load('/public/ly/hyt/RKcore/experiment/ResNet18/cifar10/selected_dataset.pth')
    if args.set == 'imagenet_dali':
        data = ImageNet()
        model = Iresnet18(num_classes=args.num_classes)
    elif args.set == 'cifar100':
        data = CIFAR100()
        model = ResNet18(num_classes=args.num_classes)
    elif args.set == 'cifar10':
        data = CIFAR10()
        model = ResNet18(num_classes=args.num_classes)
    model = set_gpu(args, model)
    logger.info(model)
    criterion = nn.CrossEntropyLoss().cuda()

    method = 'kcore'

    if method == 'random':
        data_root = '/public/MountData/dataset/cifar10'
        use_cuda = torch.cuda.is_available()
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
        )

        subset_datasets = []
        num_classes = len(train_dataset.classes)
        num_samples_per_class = len(train_dataset) // num_classes * 90 // 100
        random_indices = {}
        for class_idx in range(num_classes):
            indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_idx]
            random_indices[class_idx] = random.sample(indices, k=num_samples_per_class)
            subset_datasets += [Subset(train_dataset, random_indices[class_idx])]
        merged_dataset = torch.utils.data.ConcatDataset(subset_datasets)

        train_loader = torch.utils.data.DataLoader(
            merged_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        total_images = 0
        class_counts = [0] * num_classes
        for images, labels in train_loader:
            print(labels)
            total_images += images.shape[0]
            for label in labels:
                class_counts[label] += 1

        print(f"Total number of images in train_loader: {total_images}")
        for i in range(num_classes):
            print(f"Class {i}: {class_counts[i]}")
    elif method == 'kcore':
        # ours
        train_data = TensorDataset(selected_dataset['selected img'].cpu().pin_memory(), selected_dataset['selected lab'].cpu().pin_memory())
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
    elif method == 'Moderate-DS':
        with open('/public/ly/ICDM23/index/CIFAR10.bin', "rb") as f:
            drop_id = pickle.load(f)

        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        train_data = CIFAR10Core(root='/public/MountData/dataset/cifar10', train=True, transform=train_transform, drop_id=drop_id)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

    print('method:'.format(method))


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = cosine_lr(optimizer, args)

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # create recorder
    args.start_epoch = args.start_epoch or 0

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        scheduler(epoch, iteration=None)
        cur_lr = get_lr(optimizer)
        logger.info(f"==> CurrentLearningRate: {cur_lr}")
        train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args)
        # scheduler.step()
        acc1, acc5 = validate(data.val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                torch.save(model.state_dict(), 'pretrained_model/' + args.arch + '/' + args.set + "/scores.pt")
                logger.info(best_acc1)

if __name__ == "__main__":
    main()