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

from Data.load_data import CIFAR10
from Data.moderateds_dataset import CIFAR10Core
from args import args
import datetime
from trainer.amp_trainer_dali import train_ImageNet, validate_ImageNet
from trainer.trainer import validate, train
from utils.resnet import ResNet18
from utils.resnet_cifar import resnet20
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from utils.warmup_lr import cosine_lr


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
    selected_dataset = torch.load('/public/ly/ICDM23/experiment/clip/cifar10/selected_clip_dataset')
    #model = Iresnet18(num_classes=args.num_classes)    #used in imagenet
    model = ResNet18(num_classes=args.num_classes)
    model = set_gpu(args, model)
    logger.info(model)
    criterion = nn.CrossEntropyLoss().cuda()
    data = CIFAR10()

    method = 'kcore'

    if method == 'random':  
        # randomly split
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

        subset_lengths = [4 * len(train_dataset) // 5, len(train_dataset) // 5]
        subset_datasets = random_split(train_dataset, subset_lengths)
        train_loader = torch.utils.data.DataLoader(
            subset_datasets[1], batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
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