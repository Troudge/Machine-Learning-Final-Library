from Decision_Trees.Id3 import Id3Tree
from Decision_Trees import Id3
from Decision_Trees.tree import Node
from Linear_Regression import RegressionLearners
import networkx as nx
import matplotlib.pyplot as plt
from Ensemble_Learning import EnsembleLearners

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

dataset_bank = []
with open('../Data/bank/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_bank.append(terms)

dataset_bank_test = []
with open('../Data/bank/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_bank_test.append(terms)

dataset_concrete = []
with open('../Data/Concrete/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_concrete.append(terms)

dataset_concrete_test = []
with open('../Data/concrete/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_concrete_test.append(terms)

atr = {(0, 'Outlook'), (1, 'Temp'), (2, 'Humidity'), (3, 'Wind')}
atr2 = {(0, 'age'), (1, 'job'), (2, 'married'), (3, 'education'), (4, 'default'),
                (5, 'balance'),
                (6, 'housing'), (7, 'loan'), (8, 'contact'), (9, 'day'), (10, 'month'),
                (11, 'duration'),
                (12, 'campaign'), (13, 'pdays'), (14, 'previous'), (15, 'poutcome')}
atr3 = {(0, 'Cement'), (1, 'Slag'), (2, 'Fly ash'), (3, 'Water'), (4, 'SP'), (5, 'Course Aggr.'), (6, 'Fine Aggr.')}
# tests of the adaboost algorithm
# learner = EnsembleLearners.EnsembleLearner(dataset2, atr, 4)
# forest = learner.adaboost(1)
# print(EnsembleLearners.run_learned_forest(forest, ['R', 'C', 'N', 'W', 1], atr))



# bank_forest = bank_learner.adaboost(500)
#print(EnsembleLearners.run_forest_on_set(bank_forest, dataset_bank, atr2, 16))

#for i in range(1, 100):
#    bank_learner = EnsembleLearners.EnsembleLearner(dataset_bank, atr2, 16)
#    bank_forest_bagged = bank_learner.bagged_trees(i, 2500)
#    print(EnsembleLearners.run_forest_on_set(bank_forest_bagged, Id3.convert_numeric_set_to_boolean(dataset_bank, atr2), atr2, 16))

bank_learner = EnsembleLearners.EnsembleLearner(dataset_bank, atr2, 16)
bank_random = bank_learner.random_forest(1, 2, 2500)
print(EnsembleLearners.run_forest_on_set(bank_random, Id3.convert_numeric_set_to_boolean(dataset_bank, atr2), atr2, 16))

# An example of a decision tree stump
# decision_tree = Id3Tree(dataset2, atr, 4, 'information_gain')
# nx.draw(decision_tree.generate_id3_tree_stump().to_graph(),
#         with_labels=True, arrows=True)
# plt.show()

# concrete_learner = RegressionLearners.RegressionLearners(dataset_concrete, atr3)
# min_weights = concrete_learner.batch_gradient_descent(0.03169, 5000, 0.000001)
# print(RegressionLearners.run_weight_vec_cost_function(min_weights, dataset_concrete_test, 7))
# print("min Weights: ", min_weights)

# concrete_learner = RegressionLearners.RegressionLearners(dataset_concrete, atr3)
# min_weights = concrete_learner.stochastic_gradient_descent(0.0125, 200, 0.000001)
# print(RegressionLearners.run_weight_vec_cost_function(min_weights, dataset_concrete_test, 7))
# print("min Weights: ", min_weights)

