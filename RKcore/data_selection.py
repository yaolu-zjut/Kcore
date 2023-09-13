import numpy as np
import networkx as nx
from torch.utils.data import Dataset
import torch
from Extract_feature import calculate_cosine_similarity_matrix
from utils.kcore import roundkcore, difk,roundkcore_V1


#  python data_selection.py

def visualization():
    return


def selection(ans):
    selected_representative_id = []
    selected_hard_id = []

    id = []
    klayer = []
    rank = []

    print(len(ans))
    for kv in ans.items():
        id.append(kv[0])
        klayer.append(kv[1][0])
        rank.append(kv[1][1])

    print('Max k: {}'.format(klayer[0]))
    for i in range(len(klayer)):
        if klayer[i] == klayer[0]:
            selected_representative_id.append(id[i])

        if klayer[i] == 1:
            selected_hard_id.append(id[i])

    return selected_representative_id, selected_hard_id


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


if __name__ == "__main__":
    ########################  claculate k core  ########################
    threshold = {
        0: 0.95,
        1: 0.965,
        2: 0.95,
        3: 0.95,
        4: 0.95,
        5: 0.95,
        6: 0.955,
        7: 0.95,
        8: 0.95,
        9: 0.95
    }
 
    for class_id in range(10):
        print('Current class id: {}'.format(class_id))
        all_feats = torch.load('/public/ly/ICDM23/experiment/clip/cifar10/all_feats_clip_cifar10.pth')
        class_matrix = calculate_cosine_similarity_matrix(all_feats['feature'][class_id])  # class 0
        torch.save(class_matrix, '/public/ly/ICDM23/experiment/clip/cifar10/clip_cifar10_matrix_{}.pth'.format(class_id))
        class_matrix[class_matrix >= threshold[class_id]] = 1  # threshold # unweighted undirected graph
        class_matrix[class_matrix < threshold[class_id]] = 0  # threshold
        # cifar10:
        #  class 0: 0.95; class 1: 0.965; class 2: 0.95; class 3: 0.95; class 4: 0.95;
        #  class 5: 0.95; class 6: 0.955; class 7: 0.95; class 8: 0.95; class 9: 0.95;

        G = nx.from_numpy_matrix(class_matrix)
        safter = len(max(nx.connected_components(G), key=len))  
        print('The number of nodes in the largest connected subgraph: {}'.format(safter))

        import time

        start = time.time()
        ans = roundkcore_V1(G)  
        end = time.time()
        print('Time:{}'.format(end-start))

        id = []
        klayer = []
        rank = []

        print(len(ans))
        for kv in ans.items():
            id.append(kv[0])
            klayer.append(kv[1][0])  # in a descending sort
            rank.append(kv[1][1])  # in a descending sort

        print('Max k: {}'.format(klayer[0]))
        ranking = {'class id': class_id, 'id': id, 'k layer': klayer, 'rank': rank}
        torch.save(ranking, '/public/ly/ICDM23/experiment/clip/cifar10/ranking_clip_cifar10_{}.pth'.format(class_id))


    #############################  select data  #############################
    selected_all_img = []
    selected_all_lab = []

    num = 999
    for class_id in range(10):
        ranking = torch.load('/public/ly/ICDM23/experiment/clip/cifar10/ranking_clip_cifar10_{}.pth'.format(class_id))
        print('class id:{}'.format(ranking['class id']))

        representative_id, hard_id = ranking['id'][:num], ranking['id'][-(1000-num):]#select data
        print(len(representative_id), len(hard_id))

        all_feats = torch.load('/public/ly/ICDM23/experiment/clip/cifar10/all_feats_clip_cifar10.pth')

        selected_representative_imgs = []
        selected_representative_labs = []
        selected_hard_imgs = []
        selected_hard_labs = []

        imgs = all_feats['image']
        labs = all_feats['label']

        for i in representative_id:
            selected_representative_imgs.append(imgs[class_id][i].unsqueeze(dim=0))  # class
            selected_representative_labs.append(labs[class_id][i])  # class

        representative_labs = torch.stack(selected_representative_labs)
        representative_imgs = torch.cat(selected_representative_imgs, dim=0)

        for j in hard_id:
            selected_hard_imgs.append(imgs[class_id][j].unsqueeze(dim=0))  # class
            selected_hard_labs.append(labs[class_id][j])  # class

        hard_labs = torch.stack(selected_hard_labs)
        hard_imgs = torch.cat(selected_hard_imgs, dim=0)
        

        selected_img = torch.cat([representative_imgs, hard_imgs])
        selected_lab = torch.cat([representative_labs, hard_labs])
        print(selected_img.shape)
        print(selected_lab.shape)

        selected_all_img.append(selected_img)
        selected_all_lab.append(selected_lab)

    all_img = torch.cat(selected_all_img)
    all_lab = torch.cat(selected_all_lab)
    print(all_img.shape, all_lab.shape)
    selected_dataset = {'selected img': all_img, 'selected lab': all_lab}
    torch.save(selected_dataset, '/public/ly/ICDM23/experiment/clip/cifar10/selected_clip_dataset')
