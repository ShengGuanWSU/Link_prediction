import json
from networkx.readwrite import json_graph as jg

#origin_dataset_path = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/kb/kb_error_09_induce_04edge-unprocessed'
process_dataset_path = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/kb/kb_error_09_induce_01edge'
remove_edge_set_path = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/kb/kb_error_09_induce_0.1'

G_origin_data = json.load(open(process_dataset_path + "-G.json"))
G_origin = jg.node_link_graph(G_origin_data)
print("The origin graph edge number is")
G_origin_num_edges = G_origin.number_of_edges()
print(G_origin_num_edges)

keep_idx_train = []
for n in list(G_origin):
    if G_origin.node[n]['val'] == False and G_origin.node[n]['test'] == False:
        keep_idx_train.append(n)
# print(keep_idx_train)
for i in list(G_origin):
    # if the node is not in the "keep" list, remove it from the three objects
    if i not in keep_idx_train:
        G_origin.remove_node(i)
print("The origin graph training edge number is")
G_origin_new_num_edges = G_origin.number_of_edges()
print(G_origin_new_num_edges)


G_origin_new_edges = G_origin.edges()
G_origin_new_edges_set = set(G_origin_new_edges)

removed_edges = json.load(open(remove_edge_set_path + "-Edge.json"))
removed_edges_list =[]
for edge in removed_edges:
    removed_edges_list.append((edge[0],edge[1]))
removed_edges_set = set(removed_edges_list)

common_portion = removed_edges_set.intersection(G_origin_new_edges_set)
common_portion_len = len(common_portion)
print(common_portion_len)
removed_edges_len = len(removed_edges_set)
recovery_percentage = common_portion_len*1.00/removed_edges_len
print(recovery_percentage)

