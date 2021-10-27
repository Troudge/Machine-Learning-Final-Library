import kaggle
from Decision_Trees import Id3
from Ensemble_Learning import EnsembleLearners
# Attributes of the dataset

# age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income>50K
dataset_income_train = []
with open('../Data/Kaggle_Income/train_final.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_income_train.append(terms)

# ID,age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country
dataset_income_test = []
with open('../Data/Kaggle_Income/test_final.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_income_test.append(terms)

income_atr = {(0, 'age'), (1, 'workclass'), (2, 'fnlwgt'), (3, 'education'), (4, 'education.num'),
              (5, 'marital.status'), (6, 'occupation'), (7, 'relationship'), (8, 'race'), (9, 'sex'),
              (10, 'capital.gain'), (11, 'capital.loss'), (12, 'hours.per.week'), (13, 'native.country')}
label_col = 14

learner = EnsembleLearners.EnsembleLearner(dataset_income_train, income_atr, label_col)
income_random = learner.random_forest(100, 8, 8000)
EnsembleLearners.print_forest_outputs(income_random, dataset_income_test, income_atr)
# print(EnsembleLearners.run_forest_on_set(bank_random, Id3.convert_numeric_set_to_boolean(dataset_income_test[1:], income_atr),
#                                         income_atr, label_col))
