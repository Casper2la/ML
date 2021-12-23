import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz


import seaborn as sns

# All print() is commented out

print("3.4 Обучение, переобучение, недообучение и кросвалидация\n\n")

print("Шаг 12\n")

# Убедимся в том, что всё так происходит на самом деле.
# Скачайте тренировочный датасэт с ирисами, обучите деревья с глубиной от 1 до 100.
# Целевой переменной при обучении является переменная species.
# При этом записывайте его скор (DecisionTreeClassifier.score()) на тренировочных данных,
# и аккуратность предсказаний (accuracy_score) на тестовом датасэте.
# Затем визуализируйте зависимость скора и аккуратности предсказаний от глубины дерева и
# выберите правильную визуализацию из предложенных.
# Важно: задайте random seed прямо перед созданием дерева или укажите его в параметрах дерева (random_state=rs)
#
# np.random.seed(0)
# my_awesome_tree = DecisionTreeClassifier(...)
#
# или
#
# my_awesome_tree = DecisionTreeClassifier(random_state=0, ...)

train_iris = pd.read_csv('Lesson 3.4 data/train_iris.csv', index_col=0)
test_iris = pd.read_csv('Lesson 3.4 data/test_iris.csv', index_col=0)

X_train = train_iris.drop(['species'], axis=1)
Y_train = train_iris.species
X_test = test_iris.drop(['species'], axis=1)
Y_test = test_iris.species
iris_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
iris_tree.fit(X_train, Y_train)
# max_depth_values = range(1, 100)
# scores_data = pd.DataFrame()
# for max_depth in max_depth_values:
#     iris_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth_values, random_state=0)
#     iris_tree.fit(X_train, Y_train)
#     train_score = iris_tree.score(X_train, Y_train)
#     test_score = iris_tree.score(X_test, Y_test)
#     mean_cross_val_score = cross_val_score(iris_tree, X_train, Y_train, cv=5).mean()
#     temp_score_data = pd.DataFrame({'max_depth': [max_depth],
#                                    'train_score': [train_score],
#                                     'test_score': [test_score],
#                                     'mean_cross_val_score': [mean_cross_val_score]})
#     scores_data = scores_data.append(temp_score_data)

export_graphviz(iris_tree, out_file=None,
                feature_names=['sepal_length',  'sepal_width',  'petal_length', 'petal_width'],
                class_names=['species'], filled=True)

print(train_iris.head(), '\n')
print(test_iris.head(), '\n')
print(iris_tree, '\n')

