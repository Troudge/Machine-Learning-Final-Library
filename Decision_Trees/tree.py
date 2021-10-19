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

    def has_children(self):
        if self.children:
            return True
        return False

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

    def traverse_with_inputs(self, root, path, attributes):
        #print(root.name)
        #print(list(attributes))
        for attribute in attributes:
            #print(attribute)
            #print(attribute, attribute[1] == root.name)
            if root.name == attribute[1]:
                if root.has_children:
                    for child in root.children:
                        #print(path[attribute[0]])
                        if child.name == path[attribute[0]]:
                            new_attributes = list(attributes)
                            new_attributes.remove(attribute)
                            if child.has_children:
                                #print("traversing into:", child.children[0])
                                return self.traverse_with_inputs(child.children[0], path, new_attributes)
                else:
                    #print("root has no children. Returning: ", root.name)
                    return root.name
        #print("found no matching attribute name. Returning:", root.name)
        return root.name
