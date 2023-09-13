import datetime

import clip
import tqdm
from Data.ImageNet_dali import ImageNetDali
from Data.load_data import *
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.kcore import roundkcore
from utils.resnet_cifar import *
from utils.utils import get_logger, set_gpu
from torchvision.models.resnet import *
import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
#### step 1: python Extract_feature.py --gpu 0 --arch ResNet18 --set cifar10 --num_classes 10 --batch_size 128 --pretrained

cfgs = {
    'ResNet18': 2,
    'ResNet34': 3,
    'ResNet50': 3,
    'ResNet101': 3,
    'ResNet152': 3,
    'resnet20': 3,
    'resnet32': 5,
    'resnet44': 7,
    'resnet56': 9,
    'resnet110': 18,

}

def calculate_cosine_similarity_matrix(h_emb, eps=1e-8):
    r'''
        h_emb: (N, M) hidden representations
    '''
    # normalize
    edges_list = []
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    sim_matrix = sim_matrix.cpu().numpy()
    row, col = np.diag_indices_from(sim_matrix)  # Don not consider self-similarity
    sim_matrix[row, col] = 0

    return sim_matrix


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    if args.set == 'imagenet_dali':
        dataset = ImageNet()
    elif args.set == 'cifar10':
        dataset = CIFAR10()  # for normal training
    elif args.set == 'cifar100':
        dataset = CIFAR100()  # for normal training
    return dataset


def get_inner_feature_for_resnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    # print('cfg:', cfg)
    handle = model.layer3[cfgs[arch]-1].register_forward_hook(hook)  # here!!!
    handle_list.append(handle)
    # handle.remove()  # free memory
    return handle_list


def get_inner_feature_for_smallresnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    print('cfg:', cfg)
    handle = model.layer3[cfgs[arch]-1].register_forward_hook(hook)
    handle_list.append(handle)
    # handle.remove()  # free memory
    return handle_list


def get_model(args):
    # Note that you can train your own models using train.py
    print(f"=> Getting {args.arch}")
    if args.arch == 'ResNet18':
        model = resnet18(pretrained=True)
    elif args.arch == 'ResNet34':
        model = resnet34(pretrained=True)
    elif args.arch == 'ResNet50':
        model = resnet50(pretrained=True)
    elif args.arch == 'ResNet101':
        model = resnet101(pretrained=True)
    elif args.arch == 'ResNet152':
        model = resnet152(pretrained=True)
    elif args.arch == 'resnet20':
        model = resnet20(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet20.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet20/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet32':
        model = resnet32(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet32.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet32/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet44':
        model = resnet44(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet44.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet44/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    else:
        assert "the model has not prepared"
    # if the model is loaded from torchvision, then the codes below do not need.
    if args.set in ['cifar10', 'cifar100']:
        if args.pretrained:
            model.load_state_dict(ckpt)
        else:
            print('No pretrained model')
    else:
        print('Not mentioned dataset')
    return model


'''
# setup up:
python ASE.py --gpu 0 --arch resnet20 --set cifar10 --num_classes 10 --batch_size 256 --pretrained  --evaluate
'''

def main():  # Question1: model-dependent, Question2: effeciency
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/' + args.arch + '/' + '%s' % (args.set)):
        os.makedirs('experiment/' + args.arch + '/' + '%s' % (args.set), exist_ok=True)
    logger = get_logger('experiment/' + args.arch + '/' + '%s' % (args.set) + '/logger' + now + '.log')
    logger.info(args)

    model = get_model(args)

    logger.info(model)
    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    data = get_dataset(args)
    model.eval()
    if args.evaluate:
        if args.set in ['cifar10', 'cifar100','imagenet_dali']:
                acc1, acc5 = validate(data.val_loader, model, criterion, args)
        else:
            acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)

        logger.info(acc1)

    total_feature = []
    total_label = []
    inter_feature = []
    all_images = []

    ''' organize the real dataset '''
    images_all = [[] for c in range(args.num_classes)]
    feature_all = [[] for c in range(args.num_classes)]
    labels_all = [[] for c in range(args.num_classes)]

    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(data.train_loader), ascii=True, total=len(data.train_loader)
        ):
            if args.set in ['cifar10', 'cifar100','imagenet_dali']:
                images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)

            else:
                images = data[0]["data"].cuda(non_blocking=True)
                target = data[0]["label"].squeeze().long().cuda(non_blocking=True)

            if args.arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
                handle_list = get_inner_feature_for_smallresnet(model, hook, args.arch)
            elif args.arch in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
                handle_list = get_inner_feature_for_resnet(model, hook, args.arch)
            else:
                print('Not supported yet')

            model(images)
            all_images.append(images)
            total_feature.append(inter_feature)
            total_label.append(target)

            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()


    for c in range(len(total_label)):
        for num in range(total_label[c].shape[0]):
            labels_all[total_label[c][num]].append(total_label[c][num])
            feature_all[total_label[c][num]].append(total_feature[c][0][num].unsqueeze(dim=0).mean(dim=(2, 3)))  # pooling or just reshape
            images_all[total_label[c][num]].append(all_images[c][num].unsqueeze(dim=0))

    total_feats = []
    total_labs = []
    total_imgs = []
    for c in range(args.num_classes):
        feats = torch.cat(feature_all[c], dim=0).cuda(device=args.gpu)
        labs = torch.stack(labels_all[c])
        imgs = torch.cat(images_all[c], dim=0).cuda(device=args.gpu)
        print('class c = %d: %d real images' % (c, feats.shape[0]))
        total_feats.append(feats)
        total_labs.append(labs)
        total_imgs.append(imgs)

    # save
    all_feats = {'feature': total_feats, 'label': total_labs, 'image': total_imgs}
    save_path = 'experiment/' + args.arch + '/' + '%s' % (args.set) + '/'
    torch.save(all_feats, save_path + 'imagenette_feats_.pth')


if __name__ == "__main__":
    main()
