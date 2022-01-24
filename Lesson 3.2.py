print("3.2 Random forest\n\n")

print("Шаг 13\n")

# All print() is commented out

# Воспользуемся данными о сердечных заболеваниях и обучим на них Random Forest.
# Постройте график важности переменных для классификации и выберите среди предложенных вариантов наиболее похожий.
# В задании присутствует рандом, прогоните обучение случайного леса и построение графика несколько раз,
# чтобы увидеть изменения в важности фичей (5 самых важных обычно присутствуют в топе, просто в разном порядке).
# Чтобы получить такой же график, как в правильном варианте ответа, сделайте
# np.random.seed(0)
# rf = RandomForestClassifier(10, max_depth=5)
# Код для отрисовки важности фичей
# imp = pd.DataFrame(rf.feature_importances_, index=x_train.columns, columns=['importance'])
# imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))

# Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# импортируем данные
heart_data = pd.read_csv('Lesson 3.2 data/heart.csv')

# Выделяем из данных фичи и данные, которые нужно предсказать
X = heart_data.drop(['target'], axis=1)
y = heart_data['target']

# Рандом
np.random.seed(0)

# Создаем лес
rf = RandomForestClassifier(10, max_depth=5)

# Обучаем лес
rf.fit(X, y)

# Рисуем фичи
imp = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))

# plt.show()
