print("3.5 И на Марсе будут яблони цвести\n\n")

print("Шаг 3\n")

# All print() is commented out

# Переберите параметры с помощью GridSearchCV и обучите Random Forest на данных, указанных в предыдущем стэпе.
# Передайте в GridSearchCV модель с указанием random_state
# RandomForestClassifier(random_state=0)
# Параметры для выбора -
#     n_estimators: от 10 до 50 с шагом 10
#     max_depth: от 1 до 12 с шагом 2
#     min_samples_leaf: от 1 до 7
#     min_samples_split: от 2 до 9 с шагом 2
# Укажите cv=3. Для ускорения расчётов в GridSearchCV можно указать n_jobs=-1, чтобы использовать все процессоры.
# Какие параметры Random Forest были отобраны как наилучшие для решения на этих данных?
# Выбранное число деревьев в лесу -
# Глубина деревьев -
# Минимальное число образцов в листах -
# Минимальное число образцов для сплита -

# Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# импортируем данные
mushrooms = pd.read_csv('Lesson 3.5 data/training_mush.csv')

# Выделяем из данных фичи и данные, которые нужно предсказать
X = mushrooms.drop(['class'], axis=1)
y = mushrooms['class']

# Создаем лес
rf = RandomForestClassifier(random_state=0)

# Определяем параметры деревьев
params = {'n_estimators': range(10, 50, 10),
          'max_depth': range(1, 12, 2),
          'min_samples_leaf': range(1, 7),
          'min_samples_split': range(2, 9, 2)}

# Подбираем лучшие параметры деревьев с помощью GridSearch
search = GridSearchCV(rf, param_grid=params, cv=3, n_jobs=-1).fit(X, y)

# создаем переменную с наилучшими параметрами деревьев
best_params = search.best_params_
best_rf = search.best_estimator_
print('Выбранное число деревьев в лесу - ', best_params.get('n_estimators'), '\n',
      'Глубина деревьев - ', best_params.get('max_depth'), '\n',
      'Минимальное число образцов в листах - ', best_params.get('min_samples_leaf'), '\n',
      'Минимальное число образцов для сплита - ', best_params.get('min_samples_split'))

print("\nШаг 3\n")

# Импортируем matplotlib
import matplotlib.pyplot as plt


# Рисуем фичи
imp = pd.DataFrame(best_rf.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))

plt.show()

