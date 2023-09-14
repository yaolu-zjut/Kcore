import networkx as nx
import numpy as np


def roundkcore(G):
    G2 = G.copy()
    node_num = []  # 节点编号,度，核數
    node_mov = []
    dround = {}
    dcore = {}
    lim = len(G2)
    dorn = 0  # 删除节点否
    dt = 0  # 刪除次數
    j = 1
    step = 1
    lastk = 1
    degree_dict = G2.degree
    for point in degree_dict:
        node_num.append((point[0], point[1], -1, 0))  # 记录节点编号
    Gt = G.copy()
    while dt < lim:  # 直到网络没有节点
        degree_dict = G2.degree
        dorn = 0
        node_mov.clear()
        # nx.draw(G2, with_labels=True,node_size=3000,node_color='pink')
        for point in degree_dict:
            node_mov.append((point[0], point[1]))  # 记录节点编号
        for i in range(len(node_mov)):
            num, dgr = node_mov[i]  # 刪減中的網絡的節點與度
            if dgr <= j:
                G2.remove_node(num)  # 删除节点
                # print("刪除節點：",num)
                node_num[dt] = (num, dgr, lastk, step)
                dround[num] = step
                dcore[num] = lastk
                dt = dt + 1
                dorn = 1

        if dorn == 0:  # 上次遍历没删除 证明k值可以增加
            j = j + 1
            lastk = max(lastk, j)


        else:
            j = 1  # 上次遍历有删除 k值归零
            step += 1

    kmax = lastk
    core = []
    krankv = {}

    for i in range(kmax + 1):
        temp = {}

        for j in nx.k_shell(Gt, i, dcore).nodes:
            tempp = 0
            tempp1 = 0
            c = 0
            for k in nx.neighbors(Gt, j):
                tempp += dround[k]
                if Gt.degree(k) > 1:
                    tempp1 += 1
                c += 1
            temp[j] = dround[j] + tempp
            krankv[j] = dround[j] + tempp

        drank = sorted(temp.items(), key=lambda x: x[1], reverse=False)
        dellist = []
        for i in drank:
            dellist.append(i[0])

        core += dellist

    dellist = list(reversed(core))
    ans = {}
    for i in dellist:
        ans[i] = (dcore[i], krankv[i])
    return ans


def roundkcore_V1(G):
    dcore = nx.core_number(G)
    dround = nx.onion_layers(G)
    kmax = max(dcore.values())
    core = []
    krankv = {}
    for i in range(kmax + 1):
        temp = {}

        for j in nx.k_shell(G, i, dcore).nodes:
            tempp = 0
            tempp1 = 0
            c = 0
            for k in nx.neighbors(G, j):
                tempp += dround[k]
                if G.degree(k) > 1:
                    tempp1 += 1
                c += 1
            temp[j] = dround[j] + tempp
            krankv[j] = dround[j] + tempp

        drank = sorted(temp.items(), key=lambda x: x[1], reverse=False)
        dellist = []
        for i in drank:
            dellist.append(i[0])

        core += dellist

    dellist = list(reversed(core))
    ans = {}
    for i in dellist:
        ans[i] = (dcore[i], krankv[i])
    return ans


def difk(G):
    dcore = nx.core_number(G)
    dround = nx.onion_layers(G)
    kmax = max(dcore.values())
    ans = {}

    for k in range(kmax + 1):
        Gtemp = nx.k_core(G, k, dcore)  # subgraph of nodes with coreness greater than k
        sublen = len(Gtemp.nodes)

        for nodes in Gtemp:
            ans[nodes] = int(sublen) - int(Gtemp.degree(nodes))

    core = []
    krankv = {}
    for i in range(kmax + 1):
        temp = {}

        for j in nx.k_shell(G, i, dcore).nodes:
            temp[j] = ans[j]
        drank = sorted(temp.items(), key=lambda x: x[1], reverse=False)
        dellist = []
        for i in drank:
            dellist.append(i[0])

        core += dellist
    drank = list(reversed(core))
    delans = {}
    for i in drank:
        delans[i] = (dcore[i], ans[i])
    return delans


def difkv(G):
    dcore = nx.core_number(G)
    dround = nx.onion_layers(G)
    kmax = max(dcore.values())
    ans = {}

    for k in range(kmax + 1):
        Gtemp = nx.k_core(G, k, dcore)  # subgraph of nodes with coreness greater than k
        sublen = 2 * len(Gtemp.edges())

        for nodes in Gtemp:
            tempedgesv = 0
            for j in nx.neighbors(Gtemp, nodes):
                tempedgesv += Gtemp.degree(j)
            ans[nodes] = int(sublen) - int(tempedgesv)

    core = []
    krankv = {}
    for i in range(kmax + 1):
        temp = {}

        for j in nx.k_shell(G, i, dcore).nodes:
            temp[j] = ans[j]
        drank = sorted(temp.items(), key=lambda x: x[1], reverse=False)
        dellist = []
        for i in drank:
            dellist.append(i[0])

        core += dellist
    drank = list(reversed(core))
    delans = {}
    for i in drank:
        delans[i] = (dcore[i], ans[i])
    return delans


def difo(G):
    dcore = nx.core_number(G)
    dround = nx.onion_layers(G)
    kmax = max(dcore.values())
    omax = max(dround.values())
    ans = {}
    for k in range(omax + 1):
        temp = []
        for i in dround:
            if dround[i] >= k:
                temp.append(i)
        Gtemp = nx.subgraph(G, temp)  # subgraph of nodes with coreness greater than k
        sublen = len(Gtemp.nodes)

        for nodes in Gtemp:
            ans[nodes] = int(sublen) - int(Gtemp.degree(nodes))

    core = []

    for i in range(kmax + 1):
        temp = {}

        for j in nx.k_shell(G, i, dcore).nodes:
            temp[j] = ans[j]
        drank = sorted(temp.items(), key=lambda x: x[1], reverse=False)
        dellist = []
        for i in drank:
            dellist.append(i[0])

        core += dellist
    drank = list(reversed(core))
    delans = {}
    for i in drank:
        delans[i] = (dround[i], ans[i])
    return delans