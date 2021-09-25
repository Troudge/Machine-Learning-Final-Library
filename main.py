#!/usr/bin/env python
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import math
import tree
import statistics

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

dataset_car_test = []
with open('Data/car/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_car_test.append(terms)

dataset_bank = []
with open('Data/bank/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_bank.append(terms)

dataset_bank_test = []
with open('Data/bank/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_bank_test.append(terms)


def id3(dataset, attributes, label_col, gain_method, missing_string='?', max_depth=30):
    # graph = nx.Graph()
    depth = 0

    def run_id3(data, attribute):
        nonlocal depth
        temp_label = data[0][label_col]
        if all(row[label_col] == temp_label for row in data):
            if not attribute:
                common_node = tree.Node(get_most_common_label(data, label_col))
                return common_node
            else:
                node = tree.Node(temp_label)
                return node
        else:
            largest_gain = None
            best = ()
            for atr in attribute:
                gain = gain_method(data, atr[0], label_col, missing_string)
                if not largest_gain or gain > largest_gain:
                    largest_gain = gain
                    best = (atr[0], atr[1])

            root = tree.Node(best[1])
            for value in get_attribute_values(data, best[0]):
                child = tree.Node(value)
                root.add_child(child)
                sv = [row for row in data if row[best[0]] == value]
                if not sv:
                    child.add_child(tree.Node(get_most_common_label(sv, label_col)))
                else:
                    if depth <= max_depth:
                        depth += 1
                        child.add_child(run_id3(sv, attribute.difference((best,))))
                    else:
                        child.add_child(tree.Node(get_most_common_label(sv, label_col)))
            # depth -= 1
            return root

    # convert numerics to booleans
    for atri in attributes:
        atr_col = atri[0]
        values = [row[atr_col] for row in dataset]
        if type(values[0]) == int:
            bool_values = []
            for idx, val in enumerate(values):
                dataset[idx][atr_col] = (convert_numeric_to_bool(values, val))
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


def convert_numeric_to_bool(attribute, value):
    median = statistics.median(attribute)
    if value > median:
        return True
    else:
        return False


def get_information_gain(input_set, attribute_col, label_col, missing_string):
    attribute = get_attribute_values(input_set, attribute_col)
    entropy_sum = 0
    for value in attribute:
        if value == missing_string:
            # value = get_most_common_attribute_value(input_set, attribute_col)
            value = get_most_common_attribute_value_with_label(input_set, attribute_col, label_col)
        subset = [row[label_col] for row in input_set if row[attribute_col] == value]
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


def run_tree_on_dataset(input_tree, test_dataset, label_col, atr_names):
    result = []
    for row in test_dataset:
        result.append((row[label_col], input_tree.traverse_with_inputs(input_tree, row[:label_col], atr_names)))
    accurate_count = 0
    for tup in result:
        if tup[0] == tup[1]:
            accurate_count += 1
    # print('number of correct guesses: ', accurate_count)
    # print('number of incorrect guesses: ', len(result) - accurate_count)
    # print('Accuracy of tree:')
    print(accurate_count / len(result))
    # print(f'\n')
    return result


# Demonstration of how using the created tree works
# tree2 = id3(dataset2, {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}, 4, get_information_gain)
# print('Expected: 1 Actual: ', tree2.traverse_with_inputs(tree2, ['R', 'M', 'N', 'W'],
#                                                        {'Outlook', 'Temp', 'Humidity', 'Wind'}))
# nx.draw(tree2.to_graph(), with_labels=True, arrows=True)
# plt.show()

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
# tree creation
car_tree_basic_no_limit = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                            (5, 'safety')}, 6, get_information_gain)
car_tree_basic_depth_1 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                           (5, 'safety')}, 6, get_information_gain, max_depth=1)
car_tree_basic_depth_2 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                           (5, 'safety')}, 6, get_information_gain, max_depth=2)
car_tree_basic_depth_3 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                           (5, 'safety')}, 6, get_information_gain, max_depth=3)
car_tree_basic_depth_4 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                           (5, 'safety')}, 6, get_information_gain, max_depth=4)
car_tree_basic_depth_5 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                           (5, 'safety')}, 6, get_information_gain, max_depth=5)
car_tree_basic_depth_6 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                           (5, 'safety')}, 6, get_information_gain, max_depth=6)

print('result of id3 on the car dataset: \n', car_tree_basic_no_limit)
nx.draw(car_tree_basic_no_limit.to_graph(), with_labels=True, arrows=True)
plt.show()

print('results of standard tree no depth limit on training data:')
run_tree_on_dataset(car_tree_basic_no_limit, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of standard tree no depth limit on test data')
run_tree_on_dataset(car_tree_basic_no_limit, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of standard tree depth limit 1 on training data:')
run_tree_on_dataset(car_tree_basic_depth_1, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of standard tree depth limit 1 on test data')
run_tree_on_dataset(car_tree_basic_depth_1, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of standard tree depth limit 2 on training data:')
run_tree_on_dataset(car_tree_basic_depth_2, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of standard tree depth limit 2 on test data')
run_tree_on_dataset(car_tree_basic_depth_2, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of standard tree depth limit 3 on training data:')
run_tree_on_dataset(car_tree_basic_depth_3, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of standard tree depth limit 3 on test data')
run_tree_on_dataset(car_tree_basic_depth_3, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of standard tree depth limit 4 on training data:')
run_tree_on_dataset(car_tree_basic_depth_4, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of standard tree depth limit 4 on test data')
run_tree_on_dataset(car_tree_basic_depth_4, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of standard tree depth limit 5 on training data:')
run_tree_on_dataset(car_tree_basic_depth_5, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of standard tree depth limit 5 on test data')
run_tree_on_dataset(car_tree_basic_depth_5, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of standard tree depth limit 6 on training data:')
run_tree_on_dataset(car_tree_basic_depth_6, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of standard tree depth limit 6 on test data')
run_tree_on_dataset(car_tree_basic_depth_6, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

# same as above but with Me gain
car_tree_me_no_limit = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                         (5, 'safety')}, 6, get_me_gain)
car_tree_me_depth_1 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                        (5, 'safety')}, 6, get_me_gain, max_depth=1)
car_tree_me_depth_2 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                        (5, 'safety')}, 6, get_me_gain, max_depth=2)
car_tree_me_depth_3 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                        (5, 'safety')}, 6, get_me_gain, max_depth=3)
car_tree_me_depth_4 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                        (5, 'safety')}, 6, get_me_gain, max_depth=4)
car_tree_me_depth_5 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                        (5, 'safety')}, 6, get_me_gain, max_depth=5)
car_tree_me_depth_6 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                        (5, 'safety')}, 6, get_me_gain, max_depth=6)

print('result of id3 on the car dataset: \n', car_tree_me_no_limit)
nx.draw(car_tree_me_no_limit.to_graph(), with_labels=True, arrows=True)
plt.show()

print('results of me tree no depth limit on training data:')
run_tree_on_dataset(car_tree_me_no_limit, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of me tree no depth limit on test data')
run_tree_on_dataset(car_tree_me_no_limit, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of me tree depth limit 1 on training data:')
run_tree_on_dataset(car_tree_me_depth_1, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of me tree depth limit 1 on test data')
run_tree_on_dataset(car_tree_me_depth_1, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of me tree depth limit 2 on training data:')
run_tree_on_dataset(car_tree_me_depth_2, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of me tree depth limit 2 on test data')
run_tree_on_dataset(car_tree_me_depth_2, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of me tree depth limit 3 on training data:')
run_tree_on_dataset(car_tree_me_depth_3, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of me tree depth limit 3 on test data')
run_tree_on_dataset(car_tree_me_depth_3, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of me tree depth limit 4 on training data:')
run_tree_on_dataset(car_tree_me_depth_4, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of me tree depth limit 4 on test data')
run_tree_on_dataset(car_tree_me_depth_4, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of me tree depth limit 5 on training data:')
run_tree_on_dataset(car_tree_me_depth_5, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of me tree depth limit 5 on test data')
run_tree_on_dataset(car_tree_me_depth_5, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of me tree depth limit 6 on training data:')
run_tree_on_dataset(car_tree_me_depth_6, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of me tree depth limit 6 on test data')
run_tree_on_dataset(car_tree_me_depth_6, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

# same as above but with GINI instead
car_tree_gini_no_limit = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                           (5, 'safety')}, 6, get_gini_gain)
car_tree_gini_depth_1 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                          (5, 'safety')}, 6, get_gini_gain, max_depth=1)
car_tree_gini_depth_2 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                          (5, 'safety')}, 6, get_gini_gain, max_depth=2)
car_tree_gini_depth_3 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                          (5, 'safety')}, 6, get_gini_gain, max_depth=3)
car_tree_gini_depth_4 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                          (5, 'safety')}, 6, get_gini_gain, max_depth=4)
car_tree_gini_depth_5 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                          (5, 'safety')}, 6, get_gini_gain, max_depth=5)
car_tree_gini_depth_6 = id3(dataset_car, {(0, 'buying'), (1, 'maint'), (2, 'doors'), (3, 'persons'), (4, 'lug_boot'),
                                          (5, 'safety')}, 6, get_gini_gain, max_depth=6)

print('result of id3 on the car dataset: \n', car_tree_gini_no_limit)
nx.draw(car_tree_gini_no_limit.to_graph(), with_labels=True, arrows=True)
plt.show()

print('results of gini tree no depth limit on training data:')
run_tree_on_dataset(car_tree_gini_no_limit, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of gini tree no depth limit on test data')
run_tree_on_dataset(car_tree_gini_no_limit, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of gini tree depth limit 1 on training data:')
run_tree_on_dataset(car_tree_gini_depth_1, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of gini tree depth limit 1 on test data')
run_tree_on_dataset(car_tree_gini_depth_1, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of gini tree depth limit 2 on training data:')
run_tree_on_dataset(car_tree_gini_depth_2, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of gini tree depth limit 2 on test data')
run_tree_on_dataset(car_tree_gini_depth_2, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of gini tree depth limit 3 on training data:')
run_tree_on_dataset(car_tree_gini_depth_3, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of gini tree depth limit 3 on test data')
run_tree_on_dataset(car_tree_gini_depth_3, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of gini tree depth limit 4 on training data:')
run_tree_on_dataset(car_tree_gini_depth_4, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of gini tree depth limit 4 on test data')
run_tree_on_dataset(car_tree_gini_depth_4, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of gini tree depth limit 5 on training data:')
run_tree_on_dataset(car_tree_gini_depth_5, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of gini tree depth limit 5 on test data')
run_tree_on_dataset(car_tree_gini_depth_5, dataset_car_test, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

print('results of gini tree depth limit 6 on training data:')
run_tree_on_dataset(car_tree_gini_depth_6, dataset_car, 6,
                    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
print('results of gini tree depth limit 6 on test data')
run_tree_on_dataset(car_tree_gini_depth_6, dataset_car_test, 6, )

# Bank Section
bank_atr_set = {(0, 'age'), (1, 'job'), (2, 'married'), (3, 'education'), (4, 'default'),
                (5, 'balance'),
                (6, 'housing'), (7, 'loan'), (8, 'contact'), (9, 'day'), (10, 'month'),
                (11, 'duration'),
                (12, 'campaign'), (13, 'pdays'), (14, 'previous'), (15, 'poutcome')}
bank_atrs = ['age', 'job', 'married', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
             'duration', 'campaign', 'pdays', 'previous', 'poutcome']

# run bank stuff for default
i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_information_gain, max_depth=i)
    print(f'results of gain bank tree depth limit {i} on training data:')
    run_tree_on_dataset(bank_tree, dataset_bank, 16, bank_atrs)
    print(f'results of gain bank tree depth limit {i} on test data')
    run_tree_on_dataset(bank_tree, dataset_bank_test, 16, bank_atrs)
    i += 1
# run bank stuff for ME
i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_me_gain, max_depth=i)
    print(f'results of me bank tree depth limit {i} on training data:')
    run_tree_on_dataset(bank_tree, dataset_bank, 16, bank_atrs)
    print(f'results of me bank tree depth limit {i} on test data')
    run_tree_on_dataset(bank_tree, dataset_bank_test, 16, bank_atrs)
    i += 1
# run bank stuff for GINI
i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_gini_gain, max_depth=i)
    print(f'results of gini bank tree depth limit {i} on training data:')
    run_tree_on_dataset(bank_tree, dataset_bank, 16, bank_atrs)
    print(f'results of gini bank tree depth limit {i} on test data')
    run_tree_on_dataset(bank_tree, dataset_bank_test, 16, bank_atrs)
    i += 1

# run bank stuff for missing value
i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_information_gain, max_depth=i, missing_string='unknown')
    print('results of gini tree depth limit 6 on training data:')
    run_tree_on_dataset(bank_tree, dataset_bank, 16, bank_atrs)
    print('results of gini tree depth limit 6 on test data')
    run_tree_on_dataset(bank_tree, dataset_bank_test, 16, bank_atrs)
    i += 1
# run bank stuff for ME
i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_me_gain, max_depth=i, missing_string='unknown')
    print('results of gini tree depth limit 6 on training data:')
    run_tree_on_dataset(bank_tree, dataset_bank, 16, bank_atrs)
    print('results of gini tree depth limit 6 on test data')
    run_tree_on_dataset(bank_tree, dataset_bank_test, 16, bank_atrs)
    i += 1
# run bank stuff for GINI
i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_gini_gain, max_depth=i, missing_string='unknown')
    print('results of gini tree depth limit 6 on training data:')
    run_tree_on_dataset(bank_tree, dataset_bank, 16, bank_atrs)
    print('results of gini tree depth limit 6 on test data')
    run_tree_on_dataset(bank_tree, dataset_bank_test, 16, bank_atrs)
    i += 1

# run bank stuff for default
i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_gini_gain, max_depth=i, missing_string='unknown')
    run_tree_on_dataset(bank_tree, dataset_bank, 16, bank_atrs)
    i += 1

i = 1
while i < 17:
    bank_tree = id3(dataset_bank, bank_atr_set, 16, get_gini_gain, max_depth=i, missing_string='unknown')
    run_tree_on_dataset(bank_tree, dataset_bank_test, 16, bank_atrs)
    i += 1
