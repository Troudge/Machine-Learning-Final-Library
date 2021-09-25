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


def id3(dataset, attributes, label_col, gain_method, missing_string='?', max_depth=10):
    # graph = nx.Graph()
    depth = 0

    def run_id3(data, attribute):
        nonlocal depth
        temp_label = data[0][label_col]
        if all(row[label_col] == temp_label for row in data):
            if not attribute:
                # node = get_most_common_label(data, label)
                # graph.add_node(f"{label}_{node}")
                common_node = tree.Node(get_most_common_label(data, label_col))
                return common_node
            else:
                # graph.add_node(f"{label}_{temp_label}")
                node = tree.Node(temp_label)
                return node
        else:
            largest_gain = None
            best = ()
            for atr in attribute:
                gain = gain_method(dataset, atr[0], label_col, missing_string)
                if not largest_gain or gain > largest_gain:
                    largest_gain = gain
                    best = (atr[0], atr[1])

            # graph.add_node(best)
            root = tree.Node(best[1])
            for value in get_attribute_values(data, best[0]):
                # graph.add_node(f"{best}_{value}")
                # graph.add_edge(best, f"{best}_{value}")
                child = tree.Node(value)
                root.add_child(child)
                sv = [row for row in data if row[best[0]] == value]
                if not sv:
                    # graph.add_node(f"{best}_{get_most_common_label(sv, label)}")
                    child.add_child(tree.Node(get_most_common_label(sv, label_col)))
                else:
                    # graph.add_edge(run_id3(sv, attribute.difference((best,))), best)
                    if depth <= max_depth:
                        depth += 1
                        child.add_child(run_id3(sv, attribute.difference((best,))))
                    else:
                        child.add_child(tree.Node(get_most_common_label(sv, label_col)))
            depth -= 1
            return root

    return run_id3(dataset, attributes)


def get_most_common_label(dataset, label_col):
    count = (-1, 0)
    label_col = [row[label_col] for row in dataset]
    for lab in set(label_col):
        counter = Counter(row for row in label_col if row == lab)
        if count[1] < counter.most_common(1)[0][1]:
            count = counter.most_common(1)[0]
    return count[0]


def get_attribute_values(dataset, attribute_col):
    return set(row[attribute_col] for row in dataset)


def get_most_common_attribute_value(dataset, attribute_col):
    counter = Counter(row[attribute_col] for row in dataset)
    return counter.most_common(1)[0][0]


def get_most_common_attribute_value_with_label(dataset, attribute_col, label_col):
    count = (get_most_common_attribute_value(dataset, attribute_col), 0)
    for label in get_attribute_values(dataset, label_col):
        matching_rows = [row for row in dataset if row[label_col] == label]
        counter = Counter(row[attribute_col] for row in matching_rows)
        if len(counter) > 0:
            if count[1] < counter.most_common(1)[0][1]:
                count = counter.most_common(1)[0]
    return count[0]


def get_information_gain(input_set, attribute_col, label_col, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    entropy_sum = 0
    for value in attribute:
        if value == missing_string:
            # value = get_most_common_attribute_value(input_set, attribute_col)
            value = get_most_common_attribute_value_with_label(input_set, attribute_col, label_col)
        subset = [row[label_col] for row in input_set if row[attribute_col] == value]
        # print('subset: ', subset)
        entropy_sum += (len(subset) / len(input_set)) * get_entropy(subset)

    return get_entropy([row[label_col] for row in input_set]) - entropy_sum


def get_fractional_gain(input_set, attribute_col, label_col, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    entropy_sum = 0
    for value in attribute:
        if value == missing_string:
            # value = get_most_common_attribute_value(input_set, attribute_col)
            entropy_sum += get_fractional_entropy(input_set, value, attribute_col, label_col)
        else:
            subset = [row[label_col] for row in input_set if row[attribute_col] == value]
            # print('subset: ', subset)
            entropy_sum += (len(subset) / len(input_set)) * get_entropy(subset)

    return get_entropy([row[label_col] for row in input_set]) - entropy_sum


def get_me_gain(input_set, attribute_col, label_col, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    me_sum = 0
    for value in attribute:
        subset = [row[label_col] for row in input_set if row[attribute_col] == value]
        # print('subset: ', subset)
        me_sum += calculate_me(subset)

    return calculate_me([row[label_col] for row in input_set]) - me_sum


def get_gini_gain(input_set, attribute_col, label_col, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    gini_sum = 0
    for value in attribute:
        subset = [row[label_col] for row in input_set if row[attribute_col] == value]
        # print('subset: ', subset)
        gini_sum += calculate_gini(subset)

    return calculate_gini([row[label_col] for row in input_set]) - gini_sum


def get_entropy(input_set):
    total = 0
    for label in set(input_set):
        num = input_set.count(label)
        positive_entropy = num / len(input_set)
        # print(positive_entropy)
        negative_entropy = 1 - positive_entropy
        if negative_entropy == 0:
            return 0
        if positive_entropy == 0:
            return 1
        # print(negative_entropy)
        total += -positive_entropy * math.log2(positive_entropy) - negative_entropy * math.log2(negative_entropy)
    return total


def get_fractional_entropy(input_set, value, attribute_col, label_col):
    subset = [row for row in input_set if row[attribute_col] == value]
    c = Counter(row[label_col] for row in subset)
    negative_count = c[0]
    positive_count = c[1]
    count_sum = positive_count + negative_count
    total = count_sum / len(input_set)
    positive_entropy = positive_count / (count_sum + total)
    negative_entropy = (negative_count + total) / (count_sum + total)
    return -positive_entropy * math.log2(positive_entropy) - negative_entropy * math.log2(negative_entropy)


def calculate_me(input_set):
    total = 0
    for label in set(input_set):
        num = input_set.count(label)
        total += (num - len(input_set) / len(input_set))
        # counter = Counter(input_set)
    return total


def calculate_gini(input_set):
    counter = Counter(input_set)
    value_sum = 0
    for value in counter.values():
        value_sum += (value / len(input_set)) ** 2
    return 1 - value_sum


print(dataset_car[:5])
# Question 1 answers:
print('output of algorithm on dataset 1: \n',
      id3(dataset1, {(0, 'x1'), (1, 'x2'), (2, 'x3'), (3, 'x4')}, 4, get_information_gain))
nx.draw(id3(dataset1, {(0, 'x1'), (1, 'x2'), (2, 'x3'), (3, 'x4')}, 4, get_information_gain).to_graph(),
        with_labels=True, arrows=True)
plt.show()
# Question 2 Answers:
print('output of algorithm on dataset 2 with ME: \n',
      id3(dataset2, {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}, 4, get_me_gain))
nx.draw(id3(dataset2, {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}, 4, get_me_gain).to_graph()
        , with_labels=True, arrows=True)
plt.show()

print('output of algorithm on dataset 2 with GINI: \n',
      id3(dataset2, {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}, 4, get_gini_gain))
nx.draw(id3(dataset2, {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}, 4, get_gini_gain).to_graph()
        , with_labels=True, arrows=True)
plt.show()

# Question 3 Answers:
# print('information gain of Outlook', get_information_gain(dataset3, 0, 4, '?'))
# print('information gain of Temp', get_information_gain(dataset3, 1, 4, '?'))
# print('information gain of Humidity', get_information_gain(dataset3, 2, 4, '?'))
# print('information gain of Wind', get_information_gain(dataset3, 3, 4, '?'))

print('information gain of Outlook', get_information_gain(dataset3, 0, 4, '?'))
print('information gain of Temp', get_information_gain(dataset3, 1, 4, '?'))
print('information gain of Humidity', get_information_gain(dataset3, 2, 4, '?'))
print('information gain of Wind', get_information_gain(dataset3, 3, 4, '?'))

print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 0, 4, '?'))
print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 1, 4, '?'))
print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 2, 4, '?'))
print('information gain using fractions on set 3: ', get_fractional_gain(dataset3, 3, 4, '?'))

print('output of algorithm with fractional gain: ',
      id3(dataset3, {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}, 4, get_fractional_gain))
nx.draw(id3(dataset3, {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}, 4, get_fractional_gain).to_graph(),
        with_labels=True, arrows=True)
plt.show()

# Part 2 Car dataset Answers:

print('result of id3 on the car dataset: \n',
      id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'), (5, 'safety')}, 6,
          get_information_gain))
nx.draw(id3(dataset2, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'), (5, 'safety')}, 6,
            get_information_gain).to_graph(), with_labels=True, arrows=True)
plt.show()
