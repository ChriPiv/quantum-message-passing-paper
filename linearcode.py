import numpy as np
import networkx as nx
class LinearCode(object):

    def __init__(self, G, H):
        self.G = np.array(G)
        self.H = np.array(H)
        self.n = G.shape[1]
        self.k = G.shape[0]

    def get_codewords(self):
        def add_cw(i):
            if i == self.k-1: return [np.zeros(self.n), self.G[i]]
            ret = add_cw(i+1)
            return ret + [(self.G[i]+c)%2 for c in ret]
        return add_cw(0)

    def get_factor_graph(self):
        G = nx.Graph()
        for i in range(self.n):
            G.add_node("x"+str(i), type="variable")
            G.add_node("y"+str(i), type="output")
            G.add_edge("x"+str(i), "y"+str(i))
        for i in range(self.n-self.k):
            G.add_node("c"+str(i), type="check")
            for j in range(self.n):
                if self.H[i,j] > 0.5:
                    G.add_edge("c"+str(i), "x"+str(j))
        return G

    def get_computation_graph(self, root, height, cloner=None):
        fg = self.get_factor_graph()
        varnodes = [n for n in fg.nodes() if fg.nodes[n]["type"]=="variable"]
        G = nx.DiGraph()
        # keep track how many times each variable has been added in our digraph
        occurances = {v:0 for v in varnodes}
        # keep track how many check nodes we have added in our digraph
        num_check_nodes = 0

        max_depth = 2*height + 1
        def handle_node(node, prev, depth):
            nonlocal num_check_nodes
            if depth == max_depth: return None
            if fg.nodes[node]["type"] == "output": return None
            elif fg.nodes[node]["type"] == "variable":
                node_new = node+"_"+str(occurances[node])
                G.add_node(node_new, type="variable")
                occurances[node] += 1
            elif fg.nodes[node]["type"] == "check":
                node_new = "c"+str(num_check_nodes)
                G.add_node(node_new, type="check")
                num_check_nodes += 1
            
            descendants = [x for x in list(fg.neighbors(node)) if x != prev]
            for d in descendants:
                d_new = handle_node(d, node, depth+1)
                if d_new is not None: G.add_edge(node_new, d_new)

            if fg.nodes[node]["type"] == "variable":
                onode = node_new.replace("x", "y")
                G.add_node(onode, type="output")
                G.add_edge(node_new, onode)
            return node_new
    
        new_root = handle_node(root, None, 0)
        return G, occurances, new_root

