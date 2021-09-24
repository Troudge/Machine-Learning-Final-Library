# basic implementation of a tree structure in python
import networkx as nx


class Node:
    """a node which contains a name and a list of references to its children"""

    def __init__(self, name):
        self.name = name
        self.children = []

    def add_children(self, children):
        self.children.extend(children)

    def add_child(self, child):
        self.children.append(child)

    def __string(self, layer=0):
        return "{}{}\n{}".format('\t' * layer, self.name, "".join(
            child.__string(layer + 1) for child in self.children))

    def __str__(self):
        return self.__string()

    def to_graph(self):
        graph = nx.Graph()
        counter = 0

        def add_to_graph(root: 'Node') -> str:
            nonlocal counter
            node_name = f"{root.name} - {counter}"
            graph.add_node(node_name)
            counter += 1
            for child in root.children:
                child_name = add_to_graph(child)
                graph.add_edge(node_name, child_name)

            return node_name

        add_to_graph(self)
        return graph


class Tree:
    """A tree structure containing 1 or more nodes"""

    def __init__(self, root):
        self.root = root
