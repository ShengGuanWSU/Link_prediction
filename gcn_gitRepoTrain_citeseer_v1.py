# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:56:00 2015

@author: Timber
"""
import json
from supervisedRWfunc import *
from loadData import *
import sys
from heapq import *
from matplotlib import pyplot as plt
from added_edge import *
from dumpGraphJSON import *
import numpy as np
from loadData_citeseer import *

print "Reading data..."

# load the sanpshots of 6-30 and 12-31,
# 6-30 is a graph used as a basis of the learning process
# 12-31 provides both training data and test data
# fp = open('repos/1000_repos/snapshot-0630copy.txt', 'r')
# fp_end = open('repos/1000_repos/snapshot-1231copy.txt', 'r')


graphFile_prefix = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/citeseer/citeseer_process'
destination_dir = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/citeseer/'
datasetname = 'citeseer_process_srw'

#set random seed
seed= 321
np.random.seed(seed)
# graphFile_prefix = sys.argv[1]
G, feats, id_map, class_map = loadData_citeseer(graphFile_prefix)

nnodes = G.number_of_nodes()
edges = G.edges()


edge_feature = []
i = 0
# print(feats[edges[i][0]].shape)
# features are formed with intercept term
# luckily we have all the nodes reindex from 0 and consecutive
for i in range(len(edges)):
    # edge_feature.append([features[0][i], features[1][i]])
    temp_list = []
    temp_list = feats[edges[i][0]] + feats[edges[i][1]]
    edge_feature.append(temp_list)

#######################################
#### Training set formation ###########
#######################################

print "Forming training set..."

# compute the candidate set for future links according to source node
# train model with a set of source nodes
elig_source = []
# for i in range(len(degrees)):

for i in list(G):

    if G.degree(i) > 0 and G.node[i]['train']==True:
        # if degrees[i]>0:
        elig_source.append(i)

# pick nodes with number of future links larger than theshold
# these nodes then served as either training node or test node

'''
# visualize the graph
options = {
    'arrows': True,
    'node_color': 'blue',
    'node_size': .05,
    'line_color': 'black',
    'linewidths': 1,
    'width': 0.1,
    'with_labels': False,
    'node_shape': '.',
    'node_list': range(G.number_of_nodes())
}
nx.draw_networkx(G, **options)
plt.savefig('/Users/april/Downloads/link_prediction-master/graph/' + datasetname + 'vis.png', dpi=1024)

# print the diameter(maximum distance) of G
k = nx.connected_component_subgraphs(G)
diameter_list = []
for i in k:
    # print("Nodes in compoent.",i.nodes())
    diameter_list.append(nx.diameter(i))
'''
# print(max(diameter_list))
# Store the case to avoid the tempDset or tempLset contains two many values
Dsize_cut = 10
# Dsize_cut = int(round(sum(G.degree().values())/float(len(G))*3.76))

D_source = []
Dset_all = []
Lset_all = []

for i in range(len(elig_source)):
    # get the i's two-hop neighbors includes i
    # print i
    # two_hop_neighbors = knbrs(G, i, 3)
    # print(elig_source[-1])
    # print("current node is %d"%(elig_source[i]))
    eight_hop_neighbors_dict = nx.single_source_shortest_path_length(G, elig_source[i], cutoff=9)
    eight_hop_neighbors = set(eight_hop_neighbors_dict.keys())
    eight_hop_neighbors.remove(elig_source[i])

    # further remove one-hop neighbor
    for neighbor in G.neighbors(elig_source[i]):
        if neighbor != elig_source[i]:
            eight_hop_neighbors.remove(neighbor)

    # eight_hop_neighbors.remove(neighbor for neighbor in G.neighbors(elig_source[i]))
    candidates = eight_hop_neighbors

    # check the label of node i
    tempDset = []
    tempLset = []
    class_label = class_map[str(i)]
    for j in candidates:
        # When candidates share the same label with the source node
        # Only training part with mask whose label information is available
        if G.node[j]['train']==True and class_map[str(j)] == class_label:
            # A further semantically close logic needs to be implemented
            tempDset.append(j)

        elif G.node[j]['train']==True and class_map[str(j)] != class_label:
            tempLset.append(j)
        elif G.node[j]['train']!=True:
            print("further implement semantic closeness determination")

    # test the tempDset and tempLset at lease are all non-empty
    if len(tempDset) >= Dsize_cut and len(tempLset) >= Dsize_cut:
        #if tempLset's length >> tempDset, make their size equal
        if len(tempLset)>len(tempDset):
            keep_L_index = np.random.choice(list(range(len(tempLset))), \
                                            size=len(tempDset), replace=False)
            tempLset = [tempLset[i]for i in keep_L_index]

        Dset_all.append(tempDset)
        Lset_all.append(tempLset)
        D_source.append(elig_source[i])

        #print(tempLset[96])
'''
for i in range(len(elig_source)):
    sNeighbor = []
    for e in edges:
        if e[0] == elig_source[i]:
            sNeighbor.append(e[1])
        elif e[1] == elig_source[i]:
            sNeighbor.append(e[0])
    candidates = list(set(list(range(nnodes))) - set([elig_source[i]]) - set(sNeighbor))

    sNeighbor_end = []
    for e in edges_end:
        if e[0] == elig_source[i]:
            sNeighbor_end.append(e[1])
        elif e[1] == elig_source[i]:
            sNeighbor_end.append(e[0])
    tempDset = list(set(sNeighbor_end) - set(sNeighbor))
    if len(tempDset) >= Dsize_cut:
        tempLset = list(set(candidates) - set(tempDset))
        Dset_all.append(tempDset)
        Lset_all.append(tempLset)
        D_source.append(elig_source[i])
'''
# currently out source node should have all labels equal to [1,0]


# randomly pick nodes with current degree > 0 and number of future
# links >= Dsize_cut as the training set
# trainSize = 200
# testSize = 100
# trainSize = len(D_source)
print("length of D_source is %d" % (len(D_source)))
trainSize = min(len(D_source), 1)
testSize = len(elig_source)
# this index is the index of source nodes in D_source list
# index level selection !!!!!!!!
source_index = np.random.choice(list(range(len(D_source))), \
                                size=trainSize, replace=False)
source = []
Dset = []
Lset = []
for i in source_index:
    source.append(D_source[i])
    Dset.append(Dset_all[i])
    Lset.append(Lset_all[i])

'''

# randomly pick nodes with current degree > 0, number of future links 
# >= Dsize_cut and haven't been picked as training nodes to be test nodes
test_index = np.random.choice(list(range(len(D_source))), size=testSize, replace=False)



testSet = []
Dset_test = []
Lset_test = []
candidates_test = []


# original code
D_source_test=[]
Dset_all_test =[]
Lset_all_test=[]

for i in D_source:
    eight_hop_neighbors_dict = nx.single_source_shortest_path_length(G, i, cutoff=9)
    eight_hop_neighbors= set(eight_hop_neighbors_dict.keys())
    eight_hop_neighbors.remove(i)
    candidates = eight_hop_neighbors

    #check the label of node i
    tempDset =[]
    tempLset =[]
    class_label = class_map[str(i)]
    for j in candidates:
        #When candidates share the same label with the source node
        if class_map[str(j)]==class_label:
        #A further semantically close logic needs to be implemented
            tempDset.append(j)

        elif class_map[str(j)]!=class_label:
            tempLset.append(j)
        elif class_map[str(j)]==None:
            print("further implement semantic closeness determination")

    #test the tempDset and tempLset at lease are all non-empty
    if len(tempDset)>=Dsize_cut and len(tempLset)>0:
        Dset_all_test.append(tempDset)
        Lset_all_test.append(tempLset)
        D_source_test.append(i)


for i in test_index:
    testSet.append(D_source[i])
    Dset_test.append(Dset_all_test[i])
    Lset_test.append(Lset_all_test[i])
    candidates_test.append(Dset_all_test[i] + Lset_all_test[i])

'''

#alpha = 0.3
#######################################
#### Model training phase #############
#######################################


print "Training model..."

# set up parameters
lam = 50
offset = 0.01
alpha = 0.3
beta_init = np.ones(len(edge_feature[0])) * 2

# ff = genFeatures(nnodes, edges, edge_feature)
# trans_p = genTrans_plain(nnodes, edges, 0, 0)
# qqp = diffQ(ff, [0, 0.5, 0.5], trans_p, alpha)
# print qqp
beta_Opt = trainModel(Dset, Lset, offset, lam, nnodes, edges, edge_feature,
                      source, alpha, beta_init)

# train model direclty wtth test set, compare performance with UWRW
# beta_Opt = trainModel(Dset_test, Lset_test, offset, lam, nnodes, edges, edge_feature,
#                      testSet, alpha, beta_init)

print "Training source set:\n", source
print "\nTrained model parameters:\n", beta_Opt

##############################
####Save beta_Opt for the later quick inference########
#############################
npy_F_name = destination_dir + '/' + datasetname + '-models'
print("Saving Trained model parameters as numpy file...")
np.save(npy_F_name, beta_Opt[0])


#######################################
#### load beta_Opt[0]###########
#######################################

if os.path.exists(destination_dir+datasetname + "-models.npy"):
    model= np.load(destination_dir+datasetname + "-models.npy")
else:
    print("No model present.. Only identity features will be used.")
    model = None

#######################################
#### Test model performance ###########
#######################################

print "Evaluating model performance..."
# need to get a new set of nnodes, edges, edge_feature


# link prediction with transition matrices computed with trained parameters
ff = genFeatures(nnodes, edges, edge_feature)
# trans_srw = genTrans(nnodes, edges, ff, testSet, alpha, beta_Opt[0])
# trans_srw = genTrans(nnodes, edges, ff, testSet, alpha, [10, 10])
nodes = G.nodes()
#trans_srw = genTrans(nnodes, edges, ff, elig_source, alpha, beta_Opt[0])
trans_srw = genTrans(nnodes, edges, ff, elig_source, alpha, model)


'''
#Compute all the edges strength and get a threshold that corresponds to the smallest one
edge_prob_queue = []

for edge in edges:
    print("the edge is (%d, %d)"%(edge[0],edge[1]))
    value =trans_srw[0][edge[0]][edge[1]]
    edge_prop_entry = (value, [edge[0],edge[1]])
    heappush(edge_prob_queue, edge_prop_entry)

#Get the minimum transition value for the existing edge and use it as the threshold
minimum_edge_prop_entry = heappop(edge_prob_queue)
prop_threshold = minimum_edge_prop_entry[0]

#Scan all the labeled nodes and update their corresponding n-hop neighbors if necessary


#Scan all possible edge pairs and update the edge if above threshold
#Get the all edges set
new_edges  =[]
for n in G.node:
    for i in range(len(G.node)):
        print(i)
        if i>n:
            #contruct a potential new edge (n,i)
            if G.has_edge(n,i)==False:
                print("new edge is considered")






'''
# compute personalized PageRank for test nodes to recommend links
pgrank_srw = []
cand_pairs_srw = []
link_hits_srw = []
candidates_set = []
added_edges = []
# for i in range(len(testSet)):
num_added_edges = 0
### Here we must target all nodes
#### method for postprocessing to add edges
# added_edges=post_processing(elig_source, nnodes, trans_srw)
for i in range(len(elig_source)):
    pp = np.repeat(1.0 / nnodes, nnodes)
    curpgrank = iterPageRank(pp, trans_srw[i])
    # record the pgrank score
    pgrank_srw.append(curpgrank)

    # find the top ranking nodes in candidates set
    cand_pairs = []

    # build the candidate set for each node in elig_source
    source_set = set()
    source_set.add(elig_source[i])
    candidates_set.append([])
    candidates_set[i] = list(set(list(range(nnodes))) - source_set)

    # Print each node's candidate set according to the probability value
    for j in candidates_set[i]:
        cand_pairs.append((j, curpgrank[j]))

    cand_pairs = sorted(cand_pairs, key=lambda x: x[1], reverse=True)
    # record candidate-pagerank pairs
    cand_pairs_srw.append(cand_pairs)

    # calculate the lowest probability and its corresponding neighbors
    source_neighbors = G.neighbors(elig_source[i])


    #define a top-k for each source node
    top_k =2
    added_edge = Added_edge(elig_source[i])
    for n in range(top_k):
        if cand_pairs[n][0] in G.neighbors(elig_source[i]):
            print("There is no new edges added")
        else:
            added_edge.add_edge([elig_source[i], cand_pairs[n][0]])
            added_edge.add_edge_value(cand_pairs[n][1])
            num_added_edges += 1
    added_edges.append(added_edge)



G_data = json.load(open(graphFile_prefix + "-G.json"))
origin_G = jg.node_link_graph(G_data)

'''
num_added_edges =0
#count the num of edges
for added_edge in added_edges:
    if added_edge.edge_list!=[]:
        print("The source node id is")
        print(added_edge.source_node_id)
        print("The added edge is")
        for edge in added_edge.edge_list:
            print(edge)
            num_added_edges+=1
print(num_added_edges)
'''

# Analyze the real added edges
added_edge_threshold = len(edges)
added_edge_num = 0
for added_edge in added_edges:
    if added_edge.edge_list != []:
        if added_edge_num >= added_edge_threshold:
            break
        print("The source node id is")
        print(added_edge.source_node_id)
        print("The added edge is")
        for edge in added_edge.edge_list:
            print(edge)
            origin_G.add_edge(edge[0], edge[1])
            added_edge_num += 1
            if added_edge_num >= added_edge_threshold:
                break
    else:
        print("No edge added for this node")
        if added_edge_num >= added_edge_threshold:
            break

print("The new edge number is")
num_edges = origin_G.number_of_edges()
print(num_edges)

'''
added_edge_dict={}
for added_edge in added_edges:
    if added_edge.edge_list!=[]:
        print("The source node id is")
        print(added_edge.source_node_id)
        print("The added edge is")
        for i in range(len(added_edge.edge_list)):
            edge= added_edge.edge_list[i]
            print(edge)
            value =added_edge.value_list[i]
            added_edge_dict[[edge[0],edge[1]]]=value
'''

print("The added edge number is %d" % (num_edges - len(edges)))

dumpGraphJSON(destination_dir, datasetname, origin_G)

'''
# evaluate and compared the performance of unweighted random walk
print "Evaluating alternative models..."   

# generate unweighted transition matrices for testSet nodes
trans_uw = genTrans_plain(nnodes, edges, testSet, alpha)

# compute personalized PageRank for test nodes to recommend links
pgrank_uw = []
cand_pairs_uw = []
link_hits_uw = []
for i in range(len(testSet)):
    pp = np.repeat(1.0/nnodes, nnodes)
    curpgrank = iterPageRank(pp, trans_uw[i])
    # record the pgrank score
    pgrank_uw.append(curpgrank)

    # find the top ranking nodes in candidates set
    cand_pairs = []
    for j in candidates_test[i]:
        cand_pairs.append((j, curpgrank[j]))
    cand_pairs = sorted(cand_pairs, key = lambda x: x[1], reverse=True)
    # record candidate-pagerank pairs
    cand_pairs_uw.append(cand_pairs)

    # calculate precision of the top-Dsize_cut predicted links
    link_hits = 0    
    for j in range(Dsize_cut):
        if cand_pairs[j][0] in Dset_test[i]:
            link_hits += 1
    link_hits_uw.append(link_hits)

print "\nUW performance: ", np.mean(link_hits_uw)
'''

'''
fjson = open('repo_test_logs/git_repo_1000test.json', 'w')

beta_json = []
beta_json.append([beta_Opt[0][0], beta_Opt[0][1]])
beta_json.append(beta_Opt[1])
tempHT = beta_Opt[2]
tempHT['grad'] = [tempHT['grad'][0], tempHT['grad'][1]]
beta_json.append(tempHT)

test_log = json.dumps({'train set': source, 'test set': testSet, 
'beta': beta_json, 'SRW hit': np.mean(link_hits_srw), 'UW hit': np.mean(link_hits_uw)})
fjson.write(test_log + '\n')

fjson.close()
'''





