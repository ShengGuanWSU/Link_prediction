# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:56:00 2015
@author: Timber
"""
import json
from supervisedRWfunc import *

print "Reading data..."

# load the sanpshots of 6-30 and 12-31,
# 6-30 is a graph used as a basis of the learning process
# 12-31 provides both training data and test data
fp = open('repos/1000_repos/snapshot-0630copy.txt', 'r')
fp_end = open('repos/1000_repos/snapshot-1231copy.txt', 'r')

nnodes = 1000
degrees = [0] * nnodes
edges = []
edges_end = []
features = [[], []]
features_end = [[], []]
for line in fp:
    temp = line.strip().split(',')
    edges.append((int(temp[0]), int(temp[1])))
    features[0].append(float(temp[2]))
    features[1].append(float(temp[3]))
    degrees[int(temp[0])] += 1
    degrees[int(temp[1])] += 1

for line in fp_end:
    temp = line.strip().split(',')
    edges_end.append((int(temp[0]), int(temp[1])))
    features_end[0].append(float(temp[2]))
    features_end[1].append(float(temp[3]))

fp.close()
fp_end.close()

# normalize features
u0 = np.mean(features[0])
u1 = np.mean(features[1])
s0 = np.std(features[0])
s1 = np.std(features[1])

features[0] = map(lambda x: (x - u0) / s0, features[0])
features[1] = map(lambda x: (x - u1) / s1, features[1])

edge_feature = []
# features are formed with intercept term
for i in range(len(edges)):
    edge_feature.append([features[0][i], features[1][i]])

#######################################
#### Training set formation ###########
#######################################

print "Forming training set..."

# compute the candidate set for future links according to source node
# train model with a set of source nodes
elig_source = []
for i in range(len(degrees)):
    if degrees[i] > 0:
        elig_source.append(i)

# pick nodes with number of future links larger than theshold
# these nodes then served as either training node or test node
Dsize_cut = 20

D_source = []
Dset_all = []
Lset_all = []
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

# randomly pick nodes with current degree > 0 and number of future

#check tempDset and tempLset for different source node s1 and s2, whether they have intesection.
tempDset1 = set(Dset_all[0])
tempDset2 = set(Dset_all[1])
intersect_Dset = tempDset1.intersection(tempDset2)
#Later check these 9 nodes to test whether their p_v is the same or not

# links >= Dsize_cut as the training set
trainSize = 1
testSize = 2
# this index is the index of source nodes in D_source list
#source_index = np.random.choice(list(range(len(D_source))),
                                #size=trainSize, replace=False)
source_index =[2]
source = []
Dset = []
Lset = []
for i in source_index:
    source.append(D_source[i])
    Dset.append(Dset_all[i])
    Lset.append(Lset_all[i])

# randomly pick nodes with current degree > 0, number of future links
# >= Dsize_cut and haven't been picked as training nodes to be test nodes
test_index = np.random.choice(list(set(list(range(len(D_source)))) -
                                   set(source_index)), size=testSize, replace=False)

testSet = []
Dset_test = []
Lset_test = []
candidates_test = []

# original code

for i in test_index:
    testSet.append(D_source[i])
    Dset_test.append(Dset_all[i])
    Lset_test.append(Lset_all[i])
    candidates_test.append(Dset_all[i] + Lset_all[i])

#######################################
#### Model training phase #############
#######################################

print "Training model..."

# set up parameters
lam = 50
offset = 0.01
alpha = 0.3
beta_init = [2, 2]

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

#######################################
#### Test model performance ###########
#######################################

print "Evaluating model performance..."

# link prediction with transition matrices computed with trained parameters
ff = genFeatures(nnodes, edges, edge_feature)
trans_srw = genTrans(nnodes, edges, ff, testSet, alpha, beta_Opt[0])
# trans_srw = genTrans(nnodes, edges, ff, testSet, alpha, [10, 10])

# compute personalized PageRank for test nodes to recommend links
pgrank_srw = []
cand_pairs_srw = []
link_hits_srw = []
for i in range(len(testSet)):
    pp = np.repeat(1.0 / nnodes, nnodes)
    curpgrank = iterPageRank(pp, trans_srw[i])
    # record the pgrank score
    pgrank_srw.append(curpgrank)

    # find the top ranking nodes in candidates set
    cand_pairs = []
    for j in candidates_test[i]:
        cand_pairs.append((j, curpgrank[j]))
    cand_pairs_srw.append(cand_pairs)

print("evaluating what we want")
for common_element in intersect_Dset:
    if common_element in candidates_test[0]:
        index_common = candidates_test[0].index(common_element)
        print(common_element)
        print(candidates_test[0][index_common])
        t1 =cand_pairs_srw[0][index_common]
        index_common2 = candidates_test[1].index(common_element)
        t2 =cand_pairs_srw[1][index_common2]
        print (t1)
        print (t2)

'''
    cand_pairs = sorted(cand_pairs, key=lambda x: x[1], reverse=True)


    # record candidate-pagerank pairs
    cand_pairs_srw.append(cand_pairs)

    # calculate precision of the top-Dsize_cut predicted links
    link_hits = 0
    for j in range(Dsize_cut):
        if cand_pairs[j][0] in Dset_test[i]:
            link_hits += 1
    link_hits_srw.append(link_hits)
'''

print "\nSRW performance: ", np.mean(link_hits_srw)

# evaluate and compared the performance of unweighted random walk
print "Evaluating alternative models..."

# generate unweighted transition matrices for testSet nodes
trans_uw = genTrans_plain(nnodes, edges, testSet, alpha)

# compute personalized PageRank for test nodes to recommend links
pgrank_uw = []
cand_pairs_uw = []
link_hits_uw = []
for i in range(len(testSet)):
    pp = np.repeat(1.0 / nnodes, nnodes)
    curpgrank = iterPageRank(pp, trans_uw[i])
    # record the pgrank score
    pgrank_uw.append(curpgrank)

    # find the top ranking nodes in candidates set
    cand_pairs = []
    for j in candidates_test[i]:
        cand_pairs.append((j, curpgrank[j]))
    cand_pairs = sorted(cand_pairs, key=lambda x: x[1], reverse=True)
    # record candidate-pagerank pairs
    cand_pairs_uw.append(cand_pairs)

    # calculate precision of the top-Dsize_cut predicted links
    link_hits = 0
    for j in range(Dsize_cut):
        if cand_pairs[j][0] in Dset_test[i]:
            link_hits += 1
    link_hits_uw.append(link_hits)

print "\nUW performance: ", np.mean(link_hits_uw)

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

