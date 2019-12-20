from networkx.readwrite import json_graph as jg
import json

def dumpGraphJSON(destDirect, datasetName, graph):
    print("Dumping into JSON files...")
    # Turn graph into data
    dataG = jg.node_link_data(graph)

    # Make names
    json_G_name = destDirect + '/' + datasetName + '-G.json'


    # Dump graph into json file
    with open(json_G_name, 'w') as outputFile:
        json.dump(dataG, outputFile)

