class Added_edge:
    def __init__(self,souce_node_id):
        self.source_node_id = souce_node_id
        self.edge_list =[]
        self.value_list =[]

    def add_edge(self,edge):
        self.edge_list.append(edge)

    def add_edge_value(self, value):
        self.value_list.append(value)

