import json
from networkx.readwrite import json_graph as jg
import os
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import pickle as pkl
import sys
import scipy.sparse as sp

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def loadData_citeseer(prefix, normalize=True):
    dataset_name = 'citeseer'

    G_data = json.load(open(prefix + "-G.json"))
    G = jg.node_link_graph(G_data)
    print("The new edge number is")
    num_edges = G.number_of_edges()
    print(num_edges)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    id_map = json.load(open(prefix + "-id_map.json"))

    class_map = json.load(open(prefix + "-class_map.json"))

    #The supervised random walk must only see the training part in the graph
    #print(keep_idx_train)


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/Users/april/Downloads/GCN_detection_benchmarkFinal/GCN_detection_benchmark/gcn/data/ind.{}.{}".format(dataset_name, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/Users/april/Downloads/GCN_detection_benchmarkFinal/GCN_detection_benchmark/gcn/data/ind.{}.test.index".format(dataset_name))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # combine all training and testing features as sparse matrix
    features = sp.vstack((allx, tx)).tolil()
    # change the testing features' order, the testing instances will follow training instances
    features[test_idx_reorder, :] = features[test_idx_range, :]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    labels = np.vstack((ally, ty))
    num_nodes = labels.shape[0]

    # just to verify the features in original and features we generated are the same!!! done
    # reorganize the training node set, add another label ['train']
    train_mask = sample_mask(idx_train, num_nodes)
    i = 0
    for n in list(G):
        G.node[i]['train'] = bool(train_mask[i])
        i += 1



    featsN = np.array([feats[0]])
    for n in list(G):
        row = G.node[n]['feature']
        featsN = np.append(featsN, [row], axis=0)
    featsN = np.delete(featsN, 0, 0)

    # Index and class_map never changed after this preprocessing

    #Use PCA to reduce the feature dimension
    pca =PCA()
    featsN_pca = pca.fit_transform(featsN)
    explained_variance = pca.explained_variance_ratio_
    #print(explained_variance)
    percentage = 0.0
    threshold_per = 0.51
    kept_columns =[]
    i=0
    while percentage<threshold_per:
        percentage+=explained_variance[i]
        i+=1
        kept_columns.append(i)
    featsN_processed = featsN_pca[:,kept_columns]
    print (featsN_processed.shape)

    '''
    if normalize and not featsN_processed is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(featsN_processed)
        featsN_processed = scaler.transform(featsN_processed)
    '''


    print("transform featsN")
    featsN_list = featsN_processed.tolist()
    return G, featsN_list, id_map,class_map
