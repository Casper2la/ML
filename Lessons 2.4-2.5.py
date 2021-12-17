import pandas as pd
# All print() is commented out
print("2.4 Pandas, Dataframes \n Шаг 10")
# А теперь используем эти методы на знаменитом титаническом датасэте! Загрузите датасэт, посмотрите на датафрэйм и
# ответьте на вопросы
# Комментарий: в ответе необходимо указать количество колонок.
# Заполните пропуски
# Число колонок в представленном датафрэйме
# Число строк
# Тип float имеют колонки
# int
# object

titanic = pd.read_csv("Lessons 2.4-2.5 data/titanic.csv")
# print(titanic.info())


print("2.5 Фильтрация данных")
print("Шаг 6")
# У какой доли студентов из датасэта в колонке lunch указано free/reduced?
# Формат ответа десятичная дробь, например, 0.25

stud = pd.read_csv("Lessons 2.4-2.5 data/StudentsPerformance.csv")
# print(stud["lunch"][stud.lunch=="free/reduced"].count()/stud["lunch"].count())



print("Шаг 7")
# Как различается среднее и дисперсия оценок по предметам у групп студентов со стандартным или урезанным ланчем?
# print(stud.columns)
# print("Студенты с урезанным ланчем:", "\n",
#       "Среднее:\n",
#       "Математика:", stud["math score"][stud.lunch=="free/reduced"].mean(), "\n",
#       "Чтение:", stud["reading score"][stud.lunch=="free/reduced"].mean(), "\n",
#       "Письмо:", stud["writing score"][stud.lunch=="free/reduced"].mean(), "\n",
#       "Дисепсия:", "\n",
#       "Математика:", stud["math score"][stud.lunch == "free/reduced"].var(), "\n",
#       "Чтение:", stud["reading score"][stud.lunch == "free/reduced"].var(), "\n",
#       "Письмо:", stud["writing score"][stud.lunch == "free/reduced"].var())
# print("Студенты со стандартным ланчем:", "\n",
#       "Среднее:\n",
#       "Математика:", stud["math score"][stud.lunch == "standard"].mean(), "\n",
#       "Чтение:", stud["reading score"][stud.lunch == "standard"].mean(), "\n",
#       "Письмо:", stud["writing score"][stud.lunch == "standard"].mean(), "\n",
#       "Дисепсия:", "\n",
#       "Математика:", stud["math score"][stud.lunch == "standard"].var(), "\n",
#       "Чтение:", stud["reading score"][stud.lunch == "standard"].var(), "\n",
#       "Письмо:", stud["writing score"][stud.lunch == "standard"].var())
