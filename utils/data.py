import numpy as np
import scipy.io as sio
import torch


def load_dataset_to_torch(name, g1, g2):
    data = sio.loadmat(f'datasets/{name}.mat')

    # Load the number of nodes
    n1 = data['n1'][0, 0]
    n2 = data['n2'][0, 0]
    
    # Load the adjacency matrices
    adj1 = torch.from_numpy(data[g1].toarray()).float()
    adj2 = torch.from_numpy(data[g2].toarray()).float()

    # Load the node & edge attributes
    if f'{g1}_node_label' in data or f'{g1}_node_feat' in data:
        node_attr1_mat = data[f'{g1}_node_label'] if f'{g1}_node_label' in data else data[f'{g1}_node_feat']
        node_attr1_mat = node_attr1_mat.toarray() if not isinstance(node_attr1_mat, np.ndarray) else node_attr1_mat
        node_attr1 = torch.from_numpy(node_attr1_mat).to(torch.float64) 
    else:
        node_attr1 = None
    if f'{g2}_node_label' in data or f'{g2}_node_feat' in data:
        node_attr2_mat = data[f'{g2}_node_label'] if f'{g2}_node_label' in data else data[f'{g2}_node_feat']
        node_attr2_mat = node_attr2_mat.toarray() if not isinstance(node_attr2_mat, np.ndarray) else node_attr2_mat
        node_attr2 = torch.from_numpy(node_attr2_mat).to(torch.float64)
    else:
        node_attr2 = None

    if f'{g1}_edge_label' in data:
        edge_attr_list1 = data[f'{g1}_edge_label'].squeeze()
        edge_attr1 = []
        for edge_attr in edge_attr_list1:
            edge_attr_mat = edge_attr.toarray() if not isinstance(edge_attr, np.ndarray) else edge_attr
            edge_attr1.append(edge_attr_mat)
        edge_attr1 = torch.from_numpy(np.array(edge_attr1)).to(torch.float64)
    else:
        edge_attr1 = None
    if f'{g2}_edge_label' in data:
        edge_attr_list2 = data[f'{g2}_edge_label'].squeeze()
        edge_attr2 = []
        for edge_attr in edge_attr_list2:
            edge_attr_mat = edge_attr.toarray() if not isinstance(edge_attr, np.ndarray) else edge_attr
            edge_attr2.append(edge_attr_mat)
        edge_attr2 = torch.from_numpy(np.array(edge_attr2)).to(torch.float64)
    else:
        edge_attr2 = None

    # Load the ground truth alignments
    if 'gnd' in data:
        gnd = torch.from_numpy(data['gnd'])
    elif 'gndtruth' in data:
        gnd = torch.from_numpy(data['gndtruth'])
    else:
        gnd = torch.from_numpy(data['ground_truth'])
    
    # Load the prior alignment matrix
    H = torch.from_numpy(data['H'].toarray()) if not isinstance(data['H'], np.ndarray) else torch.from_numpy(data['H'])

    dataset = dict()
    dataset['n1'] = n1
    dataset['n2'] = n2
    dataset['adj1'] = adj1
    dataset['adj2'] = adj2
    dataset['node_attr1'] = node_attr1
    dataset['node_attr2'] = node_attr2
    dataset['edge_attr1'] = edge_attr1
    dataset['edge_attr2'] = edge_attr2
    dataset['gnd'] = gnd
    dataset['H'] = H

    return dataset


def display_dataset(dataset):
    n1 = dataset['n1']
    n2 = dataset['n2']
    adj1 = dataset['adj1']
    adj2 = dataset['adj2']
    node_attr1 = dataset['node_attr1']
    node_attr2 = dataset['node_attr2']
    edge_attr1 = dataset['edge_attr1']
    edge_attr2 = dataset['edge_attr2']
    gnd = dataset['gnd']
    H = dataset['H']

    print(f'  n1: {n1}')
    print(f'  n2: {n2}')
    print(f'  adj1: {adj1.size()}')
    print(f'  adj2: {adj2.size()}')
    if node_attr1 is not None:
        print(f'  node_attr1: {node_attr1.size()}')
    if node_attr2 is not None:
        print(f'  node_attr2: {node_attr2.size()}')
    if edge_attr1 is not None:
        print(f'  edge_attr1: {edge_attr1.size()}')
    if edge_attr2 is not None:
        print(f'  edge_attr2: {edge_attr2.size()}')
    print(f'  gnd: {gnd.size()}')
    print(f'  H: {H.size()}')
