import math
import random

from Decision_Trees import Id3
from Decision_Trees import tree


def get_error_of_tree(input_tree, test_set, label_col, attribute_names):
    error_weight_sum = 0
    result = []
    for row in test_set:
        tup = (row[label_col], input_tree.traverse_with_inputs(input_tree, row[:label_col], attribute_names))
        if tup[0] == tup[1]:
            error_weight_sum += row[label_col + 1]
        result.append(tup)
    return error_weight_sum, result


def run_learned_forest(input_forest, row, attributes):
    result = {}
    for t in input_forest:
        for a in attributes:
            if t[0].name == a[1]:
                key = t[0].traverse_with_inputs(t[0], row[a[0]], a[1])
                if key not in result:
                    result[key] = t[1]
                else:
                    result[key] += t[1]
                break
    return max(result, key=result.get)


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
        self.dataset = dataset
        self.attributes = attributes
        self.label_col = label_col
        self.attribute_names = (row[1] for row in attributes)

    def adaboost(self, T):
        # create the starting weight and give each row a weight in the set
        learned_forest = []
        starting_weight = 1 / (len(self.dataset))
        weighted_dataset = self.dataset.copy()
        for row in weighted_dataset:
            row.append(starting_weight)

        for i in range(T):
            # generate a decision stump using the weighted dataset
            id3 = Id3.Id3Tree(weighted_dataset, self.attributes, self.label_col, 'information_gain')
            stump = id3.generate_id3_tree_stump()
            # calculate the error and rows that failed and passed
            error, results = get_error_of_tree(stump, weighted_dataset, self.label_col, self.attribute_names)
            vote = 0.5 * math.log((1 - error) / error)
            learned_forest.append((stump, vote))
            weight_sum = 0
            # go through the weights of the set and update them
            for idx, row in enumerate(weighted_dataset):
                # update with a positive value if results are correct
                if results[idx][0] == results[idx][1]:
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
                        new_data.append(weighted_dataset[k])
                        break
                    num -= weighted_dataset[k][self.label_col + 1]
            weighted_dataset = new_data
        return learned_forest
