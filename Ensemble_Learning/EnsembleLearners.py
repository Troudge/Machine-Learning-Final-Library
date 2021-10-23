import math
import random

from Decision_Trees import Id3
from Decision_Trees import tree

import networkx as nx
import scipy
import matplotlib.pyplot as plt


def get_error_of_tree(input_tree, test_set, label_col, attribute_names):
    error = 0
    result = []
    for row in test_set:
        tup = (row[label_col], input_tree.traverse_with_inputs(input_tree, row[:label_col], attribute_names))
        if tup[0] != tup[1]:
            error += 1
            # error_weight_sum += row[label_col + 1]
            # print("error", error_weight_sum)
        result.append(tup)
    return error/len(test_set), result


def run_learned_forest(input_forest, row, attributes):
    result = {}
    for t in input_forest:
        for a in attributes:
            if t[0].name == a[1]:
                key = t[0].traverse_with_inputs(t[0], row, [a])
                if key not in result:
                    result[key] = t[1]
                else:
                    result[key] += t[1]
                break
    return max(result, key=lambda x: result[x])


def run_forest_on_set(input_forest, dataset, attributes, label_col):
    correct_count = 0
    for row in dataset:
        output = run_learned_forest(input_forest, row, attributes)
        if output == row[label_col]:
            correct_count += 1
    return correct_count/len(dataset)


class EnsembleLearner:
    dataset = []
    attributes = {}
    label_col = 0
    attribute_names = []

    def __init__(self, dataset, attributes, label_col):
        self.dataset = dataset.copy()
        self.attributes = attributes
        self.label_col = label_col
        self.attribute_names = (row[1] for row in attributes)

    def adaboost(self, T):
        # create the starting weight and give each row a weight in the set
        learned_forest = []
        starting_weight = 1 / (len(self.dataset))
        weighted_dataset = self.dataset.copy()
        for i in range(T):
            for row in weighted_dataset:
                row.append(starting_weight)
            #print(*weighted_dataset, sep="\n")
            # generate a decision stump using the weighted dataset
            id3 = Id3.Id3Tree(weighted_dataset, self.attributes, self.label_col, 'information_gain')
            stump = id3.generate_id3_tree_stump()
            #nx.draw(stump.to_graph(),
            #        with_labels=True, arrows=True)
            #plt.show()

            # calculate the error and rows that failed and passed
            error, results = get_error_of_tree(stump, weighted_dataset, self.label_col, self.attributes)
            if error == 0:
                error = 0.0000001
            print(error)
            vote = 0.5 * math.log((1 - error) / error)
            learned_forest.append((stump, vote))
            weight_sum = 0
            # go through the weights of the set and update them
            for idx, row in enumerate(weighted_dataset):
                # update with a positive value if results are correct
                if results[idx][0] == results[idx][1]:
                    #print(row[-1], math.exp(vote))
                    row[-1] = row[-1] * math.exp(vote)
                    weight_sum += row[-1]
                else:
                    row[-1] = row[-1] * math.exp(-vote)
                    weight_sum += row[-1]
            # divide the new weights by their sum to normalize them
            for row in weighted_dataset:
                row[-1] = row[-1] / weight_sum
            # create a new empty set with size len(weighted_dataset) and fill it out corresponding to weight values
            new_data = []
            for j in range(len(weighted_dataset)):
                num = random.random()
                for k in range(len(weighted_dataset)):
                    if num <= weighted_dataset[k][self.label_col + 1]:
                        new_data.append(weighted_dataset[k][:-1].copy())
                        break
                    num -= weighted_dataset[k][self.label_col + 1]
            weighted_dataset = new_data
        print(f"finished {T} iterations. returning")
        return learned_forest

    def bagged_trees(self, T, num_samples):
        bagged_forest = []
        for i in range(T):
            new_set = []
            for j in range(num_samples):
                new_set.append(random.choice(self.dataset).copy())
            learner = Id3.Id3Tree(new_set, self.attributes, self.label_col, "information_gain")
            id3_tree = learner.generate_id3_tree()
            bagged_forest.append((id3_tree, 0))
        return bagged_forest

    def random_forest(self, T, feature_subset_size, num_samples):
        random_forest = []
        for i in range(T):
            bootstrap_set = []
            for j in range(num_samples):
                bootstrap_set.append(random.choice(self.dataset).copy())
            atr_subset = random.sample(self.attributes, feature_subset_size)
            learner = Id3.Id3Tree(bootstrap_set, atr_subset, self.label_col, "information_gain")
            id3_tree = learner.generate_id3_tree()
            random_forest.append((id3_tree, 1))
        pass

