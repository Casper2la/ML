# А теперь используем эти методы на знаменитом титаническом датасэте! Загрузите датасэт, посмотрите на датафрэйм и
# ответьте на вопросы
#
# Комментарий: в ответе необходимо указать количество колонок.
#
# Заполните пропуски
# Число колонок в представленном датафрэйме
# Число строк
# Тип float имеют колонки
# int
# object

import pandas as pd
titanic = pd.read_csv("titanic.csv")
print("2.4 Pandas, Dataframes \n Шаг 10")
print(titanic.info())

# У какой доли студентов из датасэта в колонке lunch указано free/reduced?
# Формат ответа десятичная дробь, например, 0.25
print("2.5 Фильтрация данных")
print("Шаг 6")
stud = pd.read_csv("StudentsPerformance.csv")
print(stud["lunch"][stud.lunch=="free/reduced"].count()/stud["lunch"].count())

# Как различается среднее и дисперсия оценок по предметам у групп студентов со стандартным или урезанным ланчем?

print("Шаг 7")
print(stud.columns)
print("Студенты с урезанным ланчем:", "\n",
      "Среднее:\n",
      "Математика:", stud["math score"][stud.lunch=="free/reduced"].mean(), "\n",
      "Чтение:", stud["reading score"][stud.lunch=="free/reduced"].mean(), "\n",
      "Письмо:", stud["writing score"][stud.lunch=="free/reduced"].mean(), "\n",
      "Дисепсия:", "\n",
      "Математика:", stud["math score"][stud.lunch == "free/reduced"].var(), "\n",
      "Чтение:", stud["reading score"][stud.lunch == "free/reduced"].var(), "\n",
      "Письмо:", stud["writing score"][stud.lunch == "free/reduced"].var())
print("Студенты со стандартным ланчем:", "\n",
      "Среднее:\n",
      "Математика:", stud["math score"][stud.lunch == "standard"].mean(), "\n",
      "Чтение:", stud["reading score"][stud.lunch == "standard"].mean(), "\n",
      "Письмо:", stud["writing score"][stud.lunch == "standard"].mean(), "\n",
      "Дисепсия:", "\n",
      "Математика:", stud["math score"][stud.lunch == "standard"].var(), "\n",
      "Чтение:", stud["reading score"][stud.lunch == "standard"].var(), "\n",
      "Письмо:", stud["writing score"][stud.lunch == "standard"].var())
