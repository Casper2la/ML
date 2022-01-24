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
# search = GridSearchCV(rf, param_grid=params, cv=3, n_jobs=-1).fit(X, y)

# создаем переменную с наилучшими параметрами деревьев
# best_params = search.best_params_
# best_rf = search.best_estimator_
# print('Выбранное число деревьев в лесу - ', best_params.get('n_estimators'), '\n',
#       'Глубина деревьев - ', best_params.get('max_depth'), '\n',
#       'Минимальное число образцов в листах - ', best_params.get('min_samples_leaf'), '\n',
#       'Минимальное число образцов для сплита - ', best_params.get('min_samples_split'))

print("\nШаг 4\n")

# Импортируем matplotlib
import matplotlib.pyplot as plt


# Рисуем фичи
# imp = pd.DataFrame(best_rf.feature_importances_, index=X.columns, columns=['importance'])
# imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))

# plt.show()

print("\nШаг 5\n")

# Теперь у нас есть классификатор, определяющий какие грибы съедобные, а какие нет, испробуем его!
# Предскажите съедобность этих данных грибов и напишите в ответ число несъедобных грибов (класс равен 1).
#
# Заметьте, что для использования этого леса на новых грибах, нам нужно будет заполнить значения параметров гриба,
# часть из которых определить проще (например, цвет шляпки), а для определения части понадобится специалист-миколог.
# То есть в этом случае нам придётся самим экстрагировать признаки из объекта.
# Для зрительных признаков типа формы, цвета можно использовать более сложную модель
# (например, свёрточную нейронную сеть) и подавать на вход фотки гриба.
# И модель сама извлечёт признаки вместо того, чтобы нам описывать самим.
# Но одной фоткой тут не отделаешься - для определения запаха понадобится ещё детектор

# импортируем данные
mushrooms_test = pd.read_csv('Lesson 3.5 data/testing_mush.csv')

# Предсказываем данные на тестовой выборке
# predict = best_rf.predict(mushrooms_test)
#
# print(predict.sum())

print("\nШаг 6\n")

# Создайте confusion matrix по предсказаниям, полученным вами в прошлом уроке и правильным ответам,
# (воспользуйтесь паролем из предыдущего задания, чтобы открыть их). Выберите из предложенных вариантов правильный

# Импортируем confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Импортируем правильные ответы
answers = pd.read_csv('Lesson 3.5 data/testing_y_mush.csv')

# Строим confusion matrix
# conf_matrix = ConfusionMatrixDisplay.from_predictions(answers, predict).plot(cmap='Blues')
# plt.show()

print("\nШаг 7\n")

# На Землю нападают войска жукеров, и в их флоте присутствуют транспортники, истребители и крейсеры.
# Для борьбы с каждым типом кораблей используется свой вид оружия.
# Как аналитику из Штаба Обороны, вам поручено разработать модель, предсказывающую какие корабли участвуют в атаке,
# чтобы успешно отбить нападения на различные области планеты.
# Данных удалось собрать немного, и предсказывать придётся гораздо больший по объёму массив.
#
# Обучите модель и предскажите классы кораблей для новых поступающих данных.
# Укажите в ответе через пробел число крейсеров, транспортников и истребителей.
#
# От вашего ответа зависит судьба человечества!

# Импортируем данные для обучения
# invasion = pd.read_csv('Lesson 3.5 data/invasion.csv')
# classes = pd.DataFrame({'class': ['cruiser', 'fighter', 'transport'],
#                     'class_value': [0, 1, 2]})
# invasion = invasion.merge(classes, how='inner', on='class')
#
# Разбиваем данные на выборки для обучения
# X_train = invasion.drop(['class', 'class_value'], axis=1)
# y_train = invasion['class_value']
#
# # Создаем лес
# invasion_rf = RandomForestClassifier(random_state=0)

# Определяем параметры деревьев
# invasion_params = {'n_estimators': range(10, 50, 10),
#           'max_depth': range(2, 10, 1),
#           'min_samples_leaf': range(1, 5, 2),
#           'min_samples_split': range(2, 5, 2)}

# Подбираем лучшие параметры деревьев с помощью GridSearch
# invasion_search = GridSearchCV(invasion_rf, param_grid=invasion_params, cv=5, n_jobs=-1).fit(X_train, y_train)
#
# # создаем переменную с наилучшими параметрами деревьев
# best_invasion_rf = invasion_search.best_estimator_
# print(invasion_search.best_params_)

# импортируем тестовые данные
# invasion_test = pd.read_csv('Lesson 3.5 data/operative_information.csv')

# Предсказываем данные на тестовой выборке
# predict = best_invasion_rf.predict(invasion_test)
# predict = pd.DataFrame(predict, columns=['class_value'], index=None)
# predict_classes = predict.merge(classes, how='inner', on='class_value')
#
# print(predict_classes.groupby('class').agg({'class_value': 'count'}))

print("\nШаг 8\n")

# Какая переменная оказалась самой важной для классифицирования кораблей?

# Рисуем фичи
# imp = pd.DataFrame(best_invasion_rf.feature_importances_, index=X_train.columns, columns=['importance'])
# imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
#
# plt.show()

print("\nШаг 9\n")

# Благодаря вашим стараниям войска захватчиков были разгромлены, но война ещё не окончена!
# Вас повысили и перевели на новое направление (новые должности - новые задачи) -
# теперь нужно выявлять опасные регионы космоса, где могут находиться жукеры.
# Проанализируйте имеющиеся данные об опасности разных регионов космоса и укажите наиболее вероятные причины угрозы

# Импортируем входные данные
space = pd.read_csv('Lesson 3.5 data/space_can_be_a_dangerous_place.csv')
print(space.columns)

# Разбиваем данные на выборки для обучения
X_space_train = space.drop('dangerous', axis=1)
y_space_train = space.dangerous

# Создаем лес
space_rf = RandomForestClassifier(random_state=0)

# Определяем параметры деревьев
space_params = {'n_estimators': range(10, 50, 10),
           'max_depth': range(2, 10, 1),
           'min_samples_leaf': range(1, 5, 2),
           'min_samples_split': range(2, 5, 2)}

# Подбираем лучшие параметры деревьев с помощью GridSearch
space_search = GridSearchCV(space_rf, param_grid=space_params, cv=5, n_jobs=-1).fit(X_space_train, y_space_train)

# создаем переменную с наилучшими параметрами деревьев
best_space_rf = space_search.best_estimator_
print(space_search.best_params_)

# Рисуем фичи
imp = pd.DataFrame(best_space_rf.feature_importances_, index=X_space_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))

plt.show()