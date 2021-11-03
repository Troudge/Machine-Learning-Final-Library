def run_learned_weights(weights, dataset):
    correct = 0
    for row in dataset:
        output = 0
        for idx, value in enumerate(row[:-1]):
            output += float(value) * weights[idx]
        output += weights[-1]
        result = 1 if output > 0 else 0
        if result == int(row[-1]):
            correct += 1
    return (len(dataset) - correct)/len(dataset)


def run_voted_perceptron(weighted_weights, dataset):
    correct = 0
    for row in dataset:
        result = 0
        for weights, vote in weighted_weights:
            output = 0
            for idx, value in enumerate(row[:-1]):
                output += float(value) * weights[idx]
            output += weights[-1]
            voted_output = 1 if output > 0 else -1
            voted_output *= vote
            result += voted_output
        result = 1 if result > 0 else 0
        if result == int(row[-1]):
            correct += 1
    return (len(dataset) - correct) / len(dataset)


class PerceptronLearner:
    dataset = []
    attributes = {}
    label_col = 0
    attribute_names = []

    def __init__(self, dataset, attributes):
        self.dataset = dataset.copy()
        self.attributes = attributes

    def perceptron(self, T, learning_rate):
        weights = [0] * (len(self.attributes) + 1)
        for i in range(T):
            for row in self.dataset:
                output = 0
                # calculating the dot product
                for idx, value in enumerate(row[:-1]):
                    output += float(value) * weights[idx]
                # this is the bias
                output += weights[-1]
                result = 1 if output > 0 else 0
                for idx, value in enumerate(row[:-1]):
                    weights[idx] = weights[idx] + learning_rate * ((float(row[-1]) - result) * float(value))
                weights[-1] = weights[-1] + learning_rate * ((float(row[-1]) - result) * 1)
        return weights

    def voted_perceptron(self, T, learning_rate):
        voted_weights = []
        weights = [0] * (len(self.attributes) + 1)
        correct = 0
        for i in range(T):
            for row in self.dataset:
                output = 0
                # calculating the dot product
                for idx, value in enumerate(row[:-1]):
                    output += float(value) * weights[idx]
                # this is the bias
                output += weights[-1]
                result = 1 if output > 0 else 0
                if result == float(row[-1]):
                    correct += 1
                else:
                    for idx, value in enumerate(row[:-1]):
                        weights[idx] = weights[idx] + learning_rate * ((float(row[-1]) - result) * float(value))
                    weights[-1] = weights[-1] + learning_rate * ((float(row[-1]) - result) * 1)
                    voted_weights.append((weights.copy(), correct))
                    correct = 0
        return voted_weights

    def average_perceptron(self, T, learning_rate):
        average_weights = [0] * (len(self.attributes) + 1)
        weights = [0] * (len(self.attributes) + 1)
        for i in range(T):
            for row in self.dataset:
                output = 0
                # calculating the dot product
                for idx, value in enumerate(row[:-1]):
                    output += float(value) * weights[idx]
                # this is the bias
                output += weights[-1]
                result = 1 if output > 0 else 0
                if result != float(row[-1]):
                    for idx, value in enumerate(row[:-1]):
                        weights[idx] = weights[idx] + learning_rate * ((float(row[-1]) - result) * float(value))
                    weights[-1] = weights[-1] + learning_rate * ((float(row[-1]) - result) * 1)
                for j in range(len(average_weights)):
                    average_weights[j] = average_weights[j] + weights[j]
        return average_weights

