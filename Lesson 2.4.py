import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
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
np.random.seed(0)
train_iris = pd.read_csv('Lesson 2.4 data/train_iris.csv', index_col=0)
test_iris = pd.read_csv('Lesson 2.4 data/test_iris.csv', index_col=0)

X_train = train_iris.drop(['species'], axis=1)
Y_train = train_iris.species
X_test = test_iris.drop(['species'], axis=1)
Y_test = test_iris.species

scores_data = pd.DataFrame()
for max_depth_values in range(1, 100):
    iris_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth_values)
    iris_tree.fit(X_train, Y_train)
    train_score = iris_tree.score(X_train, Y_train)
    test_score = iris_tree.score(X_test, Y_test)
    mean_cross_val_score = cross_val_score(iris_tree, X_train, Y_train, cv=5).mean()
    temp_score_data = pd.DataFrame({'max_depth': [max_depth_values],
                                   'train_score': [train_score],
                                    'test_score': [test_score],
                                    'mean_cross_val_score': [mean_cross_val_score]})
    scores_data = scores_data.append(temp_score_data)
    scores_data[scores_data.mean_cross_val_score == scores_data.mean_cross_val_score.max()]

plt.plot(scores_data['max_depth'], scores_data['train_score'], label='train_score')
plt.plot(scores_data['max_depth'], scores_data['test_score'], label='test_score')
plt.plot(scores_data['max_depth'], scores_data['mean_cross_val_score'], label='mean_cross_val_score')
plt.ylabel('score')
plt.xlabel('depth')
# plt.legend()
# plt.show()

print("Шаг 15\n")

# Мы собрали побольше данных о котиках и собачках, и готовы обучить нашего робота их классифицировать!
# Скачайте тренировочный датасэт и  обучите на нём Decision Tree.
# После этого скачайте датасэт из задания и предскажите какие наблюдения к кому относятся.
# Введите число собачек в вашем датасэте.
# В задании допускается определённая погрешность.
# P. S.: данные в задании находятся в формате json, используйте метод pd.read_json для их прочтения

catdog = pd.read_csv('Lesson 2.4 data/dogs_n_cats.csv', index_col=0)
catdog2 = pd.read_json('Lesson 2.4 data/dataset_209691_15 (1).txt', encoding="UTF-8").set_index(['Длина'])

catdog = pd.get_dummies(catdog, dtype=np.uint8)


X_cdtr = catdog.drop(['Вид_котик', 'Вид_собачка'], axis=1)
Y_cdtr = catdog['Вид_собачка']

catdog_tree = tree.DecisionTreeClassifier(criterion='entropy')
catdog_tree.fit(X_cdtr, Y_cdtr)

# Визуализация дерева!
# tree.plot_tree(catdog_tree, feature_names=list(X_cgtr),
#      class_names=['Cat', 'Dog'],
#      filled=True)
# plt.show()

catdog2_predict = catdog_tree.predict(catdog2)
result = catdog2_predict.sum()
# print(result)
