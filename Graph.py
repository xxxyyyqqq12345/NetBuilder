import networkx as nx
import matplotlib.pyplot as plt



class Graph(object):
    def __init__(self,V=None,E=None):
        """
        V: list of nodes
        E: list of tuples
        """
        self.V=V
        self.E=E
        self.prepared=0
        self.prepare()
    def prepare(self):
        if self.V is None:
            return
        self.E_in={}
        self.E_out={}
        for vname in self.V:
            self.E_in[vname]=[]
            self.E_out[vname]=[]
        for e in self.E:
            self.E_in[e[1]]+=[e[0]]
            self.E_out[e[0]]+=[e[1]]
        self.prepared=1
    def display(self):
        G=nx.DiGraph()
        for e in self.E:
            G.add_edge(e[0],e[1])
        nx.draw(G, with_labels = True)
        plt.show()

    @property
    def has_cycle(self):
        self._init_check()
        if hasattr(self,"cycle_check"):
            return self.cycle_check
        v={}
        stack=[]
        for vname in self.V:
            v[vname]=0
        cycle=0
        for vname in self.V:
            if v[vname]==0:
                stack+=[vname]
                cycle=self._cycle_search(stack,v)
            if cycle is 1:
                break
        self.cycle_check=cycle
        return self.cycle_check
    def _cycle_search(self,stack,v):
        cycle=0
        if stack:
            for out in self.E_out[stack[-1]]:
                if out in stack:
                    cycle=1
                    break;
                else:
                    stack+=[out]
                    cycle=self._cycle_search(stack,v)
                    v[out]=1
        stack.pop(-1)
        return cycle
    def _init_check(self):
        if not self.prepared:
            raise Exception("non initialized graph")
    def get_incoming(self, vname):
        self._init_check()
        if vname not in self.E_in:
            raise Exception("non such vertices")
        return self.E_in[vname]
    def get_output(self, vname):
        self._init_check()
        if vname not in self.E_out:
            raise Exception("non such vertices")
        return self.E_out[vname]
    def get_forward_graph(self):
        self._init_check()
        if self.has_cycle:
            raise Exception("can do forward graph when cycle exist")
        forward_graph=[]
        used_node=[]
        u_node_length=-1
        while len(used_node)<len(self.V):
            if u_node_length==len(used_node):
                raise Exception("cycle detected, misinplemented cycle check?")
            u_node_length=len(used_node)
            layer=[]
            for vname,inp in self.E_in.items():
                if vname not in used_node and not [x for x in inp if x not in used_node]:
                    layer+=[vname]
            used_node+=layer
            forward_graph+=[layer]
        return forward_graph