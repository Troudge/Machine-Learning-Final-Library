# implementation of the id3 algorithm
import math
import statistics
from collections import Counter
from .tree import Node


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


class Id3Tree:
    tree = None
    dataset = []
    attributes = {}
    label_col = 0
    gain_method = ''
    missing_string = '?'
    max_depth = 30

    def __init__(self, dataset, attributes, label_col, gain_method, missing_string='?', max_depth=30):
        self.tree = None
        self.dataset = dataset
        self.attributes = attributes
        self.label_col = label_col
        if gain_method == 'information_gain':
            self.gain_method = get_information_gain
        elif gain_method == 'gini':
            self.gain_method = get_gini_gain
        elif gain_method == 'me':
            self.gain_method = get_me_gain
        elif gain_method == 'fractional':
            self.gain_method = get_fractional_gain
        else:
            self.gain_method = get_information_gain
        self.missing_string = missing_string
        self.max_depth = max_depth

    def generate_id3_tree(self):
        # graph = nx.Graph()
        depth = 0

        def run_id3(data, attribute):
            nonlocal depth
            temp_label = data[0][self.label_col]
            if all(row[self.label_col] == temp_label for row in data):
                if not attribute:
                    common_node = Node(get_most_common_label(data, self.label_col))
                    return common_node
                else:
                    node = Node(temp_label)
                    return node
            else:
                largest_gain = None
                best = ()
                for atr in attribute:
                    gain = self.gain_method(data, atr[0], self.label_col, self.missing_string)
                    if not largest_gain or gain > largest_gain:
                        largest_gain = gain
                        best = (atr[0], atr[1])

                root = Node(best[1])
                for value in get_attribute_values(data, best[0]):
                    child = Node(value)
                    root.add_child(child)
                    # the line below is new and needs to be tested
                    depth += 1
                    sv = [row for row in data if row[best[0]] == value]
                    if not sv:
                        child.add_child(Node(get_most_common_label(sv, self.label_col)))
                    else:
                        if depth <= self.max_depth:
                            depth += 1
                            child.add_child(run_id3(sv, attribute.difference((best,))))
                        else:
                            child.add_child(Node(get_most_common_label(sv, self.label_col)))
                # depth -= 1
                return root

        # convert numerics to booleans
        for atri in self.attributes:
            atr_col = atri[0]
            values = [row[atr_col] for row in self.dataset]
            if type(values[0]) == int:
                bool_values = []
                for idx, val in enumerate(values):
                    self.dataset[idx][atr_col] = (convert_numeric_to_bool(values, val))
        self.tree = run_id3(self.dataset, self.attributes)
        return self.tree

    def generate_id3_tree_stump(self):
        def run_id3(data, attribute):
            temp_label = data[0][self.label_col]
            if all(row[self.label_col] == temp_label for row in data):
                if not attribute:
                    common_node = Node(get_most_common_label(data, self.label_col))
                    return common_node
                else:
                    node = Node(temp_label)
                    return node
            else:
                largest_gain = None
                best = ()
                for atr in attribute:
                    gain = get_gini_gain(data, atr[0], self.label_col, self.missing_string)
                    if not largest_gain or gain > largest_gain:
                        largest_gain = gain
                        best = (atr[0], atr[1])

                root = Node(best[1])
                for value in get_attribute_values(data, best[0]):
                    child = Node(value)
                    root.add_child(child)
                    sv = [row for row in data if row[best[0]] == value]
                    if not sv:
                        child.add_child(Node(get_most_common_label(sv, self.label_col)))
                    else:
                        child.add_child(Node(get_most_common_label(sv, self.label_col)))
                return root

        # convert numerics to booleans
        for atri in self.attributes:
            atr_col = atri[0]
            values = [row[atr_col] for row in self.dataset]
            if type(values[0]) == int:
                bool_values = []
                for idx, val in enumerate(values):
                    self.dataset[idx][atr_col] = (convert_numeric_to_bool(values, val))
        self.tree = run_id3(self.dataset, self.attributes)
        return self.tree

    def run_tree_on_dataset(self, test_dataset, atr_names):
        if self.tree is None:
            self.tree = self.generate_id3_tree()
        result = []
        for row in test_dataset:
            result.append(
                (row[self.label_col], self.tree.traverse_with_inputs(self.tree, row[:self.label_col], atr_names)))
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
