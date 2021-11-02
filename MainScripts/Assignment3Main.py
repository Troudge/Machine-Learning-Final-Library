from Perceptron import PerceptronLearner

dataset_banknote = []
with open('../Data/bank-note/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_banknote.append(terms)

dataset_banknote_test = []
with open('../Data/bank-note/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_banknote_test.append(terms)

atr = {(0, 'variance'), (1, 'skewness'), (2, 'curtosis'), (3, 'entropy')}
learner = PerceptronLearner.PerceptronLearner(dataset_banknote, atr)
weights = learner.perceptron(10, 0.3)
print(PerceptronLearner.run_learned_weights(weights, dataset_banknote_test))
voted_weights = learner.voted_perceptron(10, 0.3)
print(PerceptronLearner.run_voted_perceptron(voted_weights, dataset_banknote_test))
average_weights = learner.average_perceptron(10, 0.3)
print(PerceptronLearner.run_learned_weights(average_weights, dataset_banknote_test))
