import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import math

# All print() is commented out

print("2.7 Практика, Scikit-learn, fit, predict, you are awesome\n\n")

print("Шаг 3\n")

# Скачайте набор данных с тремя переменными: sex, exang, num.
# Представьте, что при помощи дерева решений мы хотим классифицировать
# есть или нет у пациента заболевание сердца (переменная num), основываясь на двух признаках:
# пол(sex) и наличие / отсутсвие стенокардии(exang).
# Обучите дерево решений на этих данных, используйте entropy в качестве критерия.
# Укажите, чему будет равняться значение Information Gain для переменной, которая будет помещена в корень дерева.
# В ответе необходимо указать число с точностью 3 знака после запятой.

train_data = pd.read_csv('Lesson 2.7 data/train_data_tree.csv', index_col=0)

X_train = train_data.drop(['num'], axis=1)
Y_train = train_data.num
exang_tree = tree.DecisionTreeClassifier(criterion='entropy')
exang_tree.fit(X_train, Y_train)

def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = np.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)

    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            # use log from math and set base to 2
            entropy += prob * math.log(prob, 2)

    return -entropy

def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a data set, column to split on, and target
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])

    # Find the unique values in the column
    values = data[split_name].unique()

    # Make two subsets of the data, based on the unique values
    left_split = data[data[split_name] == values[0]]
    right_split = data[data[split_name] == values[1]]

    # Loop through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0])
        to_subtract += prob * calc_entropy(subset[target_name])

    # Return information gain
    return original_entropy - to_subtract

# print("Энтропия по фиче sex - ", calc_entropy(train_data.index))
# print("Энтропия по фиче exang - ", calc_entropy(train_data.exang))
# print("information gain - ", round(calc_information_gain(train_data, 'exang', 'num'), 3))
#
# tree.plot_tree(exang_tree, feature_names=list(X_train),
#      class_names=['0', '1'],
#      filled=True)
# plt.show()

print("Шаг 6\n")

# Теперь, создав дерево, давайте обучим его и попробуем что-нибудь предсказать!
# Для начала опробуем наше дерево на классическом наборе iris, где собраны данные о длине,
# ширине чашелистиков и лепестков ирисов и их принадлежности к виду. В sklearn он уже встроен, что довольно удобно.
# Итак, вам даны 2 numpy эррея с измеренными признаками ирисов и их принадлежностью к виду.
# Сначала попробуем примитивный способ с разбиением данных на 2 датасэта.
# Используйте функцию train_test_split для разделения имеющихся данных на тренировочный и тестовый наборы данных,
# 75% и 25% соответственно.
# Затем создайте дерево dt с параметрами по умолчанию и обучите его на тренировочных данных,
# а после предскажите классы, к которым принадлежат данные из тестовой выборки,
# сохраните результат предсказаний в переменную predicted.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X, y, test_size=0.25, random_state=42)
dt = DecisionTreeClassifier()
dt.fit(X_iris_train, y_iris_train)
predicted = dt.predict(X_iris_test)

print("Шаг 10\n")

# Одно дерево - хорошо, но где гарантии, что оно является лучшим, или хотя бы близко к нему?
# Одним из способов найти более-менее оптимальный набор параметров дерева
# является перебор множества деревьев с разными параметрами и выбор подходящего.
# Для этой цели существует класс GridSearchCV, перебирающий каждое из сочетаний параметров среди заданных для модели,
# обучающий её на данных и проводящих кросс-валидацию.
# После этого в аттрибуте .best_estimator_ храниться модель с лучшими параметрами.
# Это применимо не только к деревьям, но и к другим моделям sklearn.
# Теперь задание - осуществите перебор всех деревьев на данных ириса по следующим параметрам:
# максимальная глубина - от 1 до 10 уровней
# минимальное число проб для разделения - от 2 до 10
# минимальное число проб в листе - от 1 до 10
# и сохраните в переменную best_tree лучшее дерево.
# Переменную с GridSearchCV назовите search

from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier()

param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
search = GridSearchCV(dt, param_grid=param_grid).fit(X, y)
best_tree = search.best_estimator_

print("Шаг 11\n")

# Чем больше данных, сложность модели и число её параметров, тем дольше будет вестись поиск GridSearchCV.
# Однако бывают случаи, когда модель нужна здесь и сейчас, и для этого есть RandomizedSearchCV!
# Пробегаясь по рандомной подвыборке параметров,
# он ищет наиболее хорошую модель и делает это быстрее полного перебора параметров,
# хотя и может пропустить оптимальные параметры.
# Здесь можно посмотреть на сравнение этих поисков.
# Осуществим поиск по тем же параметрам что и в предыдущем задании с помощью RandomizedSearchCV
# максимальная глубина - от 1 до 10 уровней
# минимальное число проб для разделения - от 2 до 10
# минимальное число проб в листе - от 1 до 10
# Cохраните в переменную best_tree лучшее дерево. Переменную с RandomizedSearchCV назовите search

from sklearn.model_selection import RandomizedSearchCV

dt = DecisionTreeClassifier()

param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
search = RandomizedSearchCV(dt, param_distributions=param_grid).fit(X, y)
best_tree = search.best_estimator_

print("Шаг 12\n")

# Воспользуемся изученными приёмами и попредсказываем!
# Даны 2 датасэта, к которым вы можете обращаться:
#     train - размеченный с известными правильным ответами (хранятся в колонке y)
#     test - набор, где нужно предсказать их
# Найдите дерево с наиболее подходящими параметрами с помощью GridSearchCV
# и предскажите с его помощью ответы ко 2-ому сэту! Границы параметров как раньше:
# максимальная глубина - от 1 до 10 уровней
# минимальное число проб для разделения - от 2 до 10
# минимальное число проб в листе - от 1 до 10
# Названия переменных тоже:лучшее дерево - best_tree, GridSearchCV - search, а предсказания - predictions

X_train = train.drop(['y'], axis=1)
y_train = train.y

dt = DecisionTreeClassifier()

param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
search = GridSearchCV(dt, param_grid=param_grid).fit(X_train, y_train)
best_tree = search.best_estimator_
predictions = best_tree.predict(test)

print("Шаг 13\n")

# При классификации модель может допускать ошибки, присваивая наблюдению неверный класс.
# Существуют различные метрики оценки качества предсказаний, которые базируются на 4-ёх параметрах -
# true positive, false positive, false negative и true negative,
# соответствующих тому какой класс был присвоен наблюдениям каждого из классов.
# Матрицу из 4-ёх (в случае бинарной классификации) этих параметров называют confusion matrix.
# В sklearn можно её удобно получить с помощью функции confusion_matrix.
# Вам даны 2 эррея с истинными классами наблюдений и предсказанными - y и predictions.
# Получите по ним confusion matrix и поместите её в переменную conf_matrix.

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y, predictions)