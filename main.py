import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import math
import tree

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
dataset1 = [[0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0]]

dataset2 = [['S', 'H', 'H', 'W', 0],
            ['S', 'H', 'H', 'S', 0],
            ['O', 'H', 'H', 'W', 1],
            ['R', 'M', 'H', 'W', 1],
            ['R', 'C', 'N', 'W', 1],
            ['R', 'C', 'N', 'S', 0],
            ['O', 'C', 'N', 'S', 1],
            ['S', 'M', 'H', 'W', 0],
            ['S', 'C', 'N', 'W', 1],
            ['R', 'M', 'N', 'W', 1],
            ['S', 'M', 'N', 'S', 1],
            ['O', 'M', 'H', 'S', 1],
            ['O', 'H', 'N', 'W', 1],
            ['R', 'M', 'H', 'S', 0]]

dataset3 = [['S', 'H', 'H', 'W', 0],
            ['S', 'H', 'H', 'S', 0],
            ['O', 'H', 'H', 'W', 1],
            ['R', 'M', 'H', 'W', 1],
            ['R', 'C', 'N', 'W', 1],
            ['R', 'C', 'N', 'S', 0],
            ['O', 'C', 'N', 'S', 1],
            ['S', 'M', 'H', 'W', 0],
            ['S', 'C', 'N', 'W', 1],
            ['R', 'M', 'N', 'W', 1],
            ['S', 'M', 'N', 'S', 1],
            ['O', 'M', 'H', 'S', 1],
            ['O', 'H', 'N', 'W', 1],
            ['R', 'M', 'H', 'S', 0],
            ['?', 'M', 'N', 'W', 1]]

dataset_car = []
with open('Data/car/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_car.append(terms)


def id3(dataset, attributes, label, gain_method, missing_string='?', max_depth=6):
    # graph = nx.Graph()
    depth = 0

    def run_id3(data, attribute):
        nonlocal depth
        temp_label = data[0][label]
        if all(row[label] == temp_label for row in data):
            if not attribute:
                # node = get_most_common_label(data, label)
                # graph.add_node(f"{label}_{node}")
                common_node = tree.Node(get_most_common_label(data, label))
                return common_node
            else:
                # graph.add_node(f"{label}_{temp_label}")
                node = tree.Node(temp_label)
                return node
        else:
            largest_gain = 0
            best = 0
            for atr in attribute:
                gain = gain_method(dataset, atr, label, missing_string)
                if gain > largest_gain:
                    largest_gain = gain
                    best = atr

            # graph.add_node(best)
            root = tree.Node(best)
            for value in get_attribute_values(data, best):
                # graph.add_node(f"{best}_{value}")
                # graph.add_edge(best, f"{best}_{value}")
                child = tree.Node(value)
                root.add_child(child)
                sv = [row for row in data if row[best] == value]
                if not sv:
                    # graph.add_node(f"{best}_{get_most_common_label(sv, label)}")
                    child.add_child(tree.Node(get_most_common_label(sv, label)))
                else:
                    # graph.add_edge(run_id3(sv, attribute.difference((best,))), best)
                    if depth < max_depth:
                        depth += 1
                        child.add_child(run_id3(sv, attribute.difference((best,))))
                    else:
                        child.add_child(get_best(sv, attribute.difference((best,)), label))
            return root

    return run_id3(dataset, attributes)


def get_best(input_set, attributes, label):
    for attribute in attributes:
        return tree.Node(get_most_common_attribute_value_with_label(input_set, attribute, label, 1))


def get_most_common_label(dataset, label):
    label_col = [row[label] for row in dataset]
    if sum(label_col) > len(label_col) / 2:
        return 1
    return 0


def get_attribute_values(dataset, attribute_col):
    return set(row[attribute_col] for row in dataset)


def get_most_common_attribute_value(dataset, attribute_col):
    counter = Counter(row[attribute_col] for row in dataset)
    return counter.most_common(1)[0][0]


def get_most_common_attribute_value_with_label(dataset, attribute_col, label_col, label):
    matching_rows = [row for row in dataset if row[label_col] == label]
    counter = Counter(row[attribute_col] for row in matching_rows)
    return counter.most_common(1)[0][0]


def get_information_gain(input_set, attribute_col, label, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    entropy_sum = 0
    for value in attribute:
        if value == missing_string:
            value = get_most_common_attribute_value(input_set, attribute_col)
        subset = [row[label] for row in input_set if row[attribute_col] == value]
        # print('subset: ', subset)
        entropy_sum += (len(subset) / len(input_set)) * get_entropy(subset)

    return get_entropy([row[label] for row in input_set]) - entropy_sum


def get_fractional_gain(input_set, attribute_col, label, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    entropy_sum = 0
    for value in attribute:
        if value == missing_string:
            # value = get_most_common_attribute_value(input_set, attribute_col)
            entropy_sum += get_fractional_entropy(input_set, value, attribute_col, label)
        else:
            subset = [row[label] for row in input_set if row[attribute_col] == value]
            # print('subset: ', subset)
            entropy_sum += (len(subset) / len(input_set)) * get_entropy(subset)

    return get_entropy([row[label] for row in input_set]) - entropy_sum


def get_me_gain(input_set, attribute_col, label, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    me_sum = 0
    for value in attribute:
        subset = [row[label] for row in input_set if row[attribute_col] == value]
        # print('subset: ', subset)
        me_sum += calculate_me(subset)

    return calculate_me([row[label] for row in input_set]) - me_sum


def get_gini_gain(input_set, attribute_col, label, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    gini_sum = 0
    for value in attribute:
        subset = [row[label] for row in input_set if row[attribute_col] == value]
        # print('subset: ', subset)
        gini_sum += calculate_gini(subset)

    return calculate_gini([row[label] for row in input_set]) - gini_sum


def get_entropy(input_set):
    # print('input set:', input_set)
    # print('sum of input set: ', sum(input_set))
    positive_entropy = sum(input_set) / len(input_set)
    # print(positive_entropy)
    negative_entropy = 1 - positive_entropy
    if negative_entropy == 0:
        return 0
    if positive_entropy == 0:
        return 1
    # print(negative_entropy)
    return -positive_entropy * math.log2(positive_entropy) - negative_entropy * math.log2(negative_entropy)


def get_fractional_entropy(input_set, value, attribute_col, label):
    subset = [row for row in input_set if row[attribute_col] == value]
    c = Counter(row[label] for row in subset)
    negative_count = c[0]
    positive_count = c[1]
    count_sum = positive_count + negative_count
    total = count_sum / len(input_set)
    positive_entropy = positive_count / (count_sum + total)
    negative_entropy = (negative_count + total) / (count_sum + total)
    return -positive_entropy * math.log2(positive_entropy) - negative_entropy * math.log2(negative_entropy)


def calculate_me(input_set):
    counter = Counter(input_set)
    return len(input_set) - counter[1] / len(input_set)


def calculate_gini(input_set):
    counter = Counter(input_set)
    value_sum = 0
    for value in counter.values():
        value_sum += (value / len(input_set)) ** 2
    return 1 - value_sum


# print('dataset 1 entropy: ', get_entropy([row[-1] for row in dataset1]))
# print('dataset 2 entropy: ', get_entropy([row[-1] for row in dataset2]))
# print('dataset 2 Outlook = Sunny: ', get_entropy([row[-1] for row in dataset2 if row[0] == 'S']))
# print('information gain of Outlook', get_information_gain(dataset2, 0, 4, ''))
# print('information gain of Humidity', get_information_gain(dataset2, 2, 4, ''))
print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 0, 4, '?'))
print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 1, 4, '?'))
print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 2, 4, '?'))
print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 3, 4, '?'))

# print('most common label for Outlook', get_most_common_attribute_value(dataset2, 0))
# print('most common label for Outlook', get_most_common_attribute_value_with_label(dataset2, 0, 4, 1))

# print(id3(dataset2, {0, 1, 2, 3}, 4, get_information_gain))
print('result of id3 with missing value: ', id3(dataset3, {0, 1, 2, 3}, 4, get_fractional_gain))
nx.draw(id3(dataset2, {0, 1, 2, 3}, 4, get_fractional_gain).to_graph(), with_labels=True, arrows=True)
# nx.draw(id3(dataset2, {0, 1, 2, 3}, 4, get_information_gain).to_graph(), with_labels=True, arrows=True)
plt.show()
