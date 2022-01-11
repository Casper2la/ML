import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

# All print() is commented out

print("2.5 Последний джедай или метрики качества модели\n\n")

print("Шаг 13\n")

# Поупражняемся в расчётах precision. В задании даны переменные,
# содержащие информацию о песнях и артистах - X_train, y_train, X_test и y_test.
# Исходно в датасэтах содержались тексты песен, но Decision Tree работает с категориальными и числовыми переменными,
# а текст это... текст. Поэтому его необходимо преобразовать в понятную для модели форму.
# В данном случае для каждой песни просто посчитаны длина и количество некоторых знаков пунктуации.
# Обучите модель на тренировочных данных, предскажите авторов для тестовых и поместите в переменную predictions.
# Затем посчитайте precision score на предсказаниях и y_test, укажите параметр average='micro',
# и сохраните результат в переменную precision.
# Если он будет недостаточно высок, потюньте немного модель.
# Исходные данные взяты отсюда, слегка процессированные можно взять здесь
# (исходные колонки типа жанра, года были выкинуты в задании)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
precision = precision_score(y_test, predictions, average='micro')