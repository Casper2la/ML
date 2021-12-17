import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# All print() is commented out
print("2.8 Практические задания: Pandas\n\n")

print("Шаг 2\n")

# Любым удобным для вас способом создайте dataframe c именем my_data, в котором две колонки c именами
# (type - строки, value - целые числа) и четыре наблюдения в каждой колонке:
# type value
# A       10
# A       14
# B       12
# B       23
data = {"type": ["A", "A", "B", "B"], "value": [10, 14, 12, 23]}
my_data = pd.DataFrame(data, columns=["type", "value"])
# print(my_data)

print("Шаг 3\n")

# Особенно важный навык при работе с данными - это умение быстро и эффективно отбирать нужные вам колонки или строки.
# Начнем с простого, в dataframe с именем my_stat сохранено 20 строк и четыре колонки (V1, V2, V3, V4):
# В переменную с именем subset_1 сохраните только первые 10 строк и только 1 и 3 колонку.
# В переменную с именем subset_2 сохраните все строки кроме 1 и 5 и только 2 и 4 колонку.
# Помните, что нумерация индексов строк и колонок начинается с 0.
# Обратите внимание, получившиеся subset_1 и subset_2 - тоже должны быть dataframe.
# Вы можете скачать набор данных, которые нам также пригодятся в следующих заданиях,
# и потренироваться у себя на компьютере. Чтобы считать данные при помощи pandas, используйте функцию read_csv.
# Важно понимать, в чем разница между pandas loc и iloc. Как отобрать все строки кроме указанных?
# Умение искать ответы поможет вам на начальных этапах знакомства с pandas!

my_stat = pd.read_csv('Lesson 2.8 data/my_stat.csv')
subset_1_3 = my_stat.iloc[0:10, [0, 2]]
subset_2_3 = pd.concat([my_stat.iloc[1:4, [1, 3]], my_stat.iloc[5:, [1, 3]]])
# print(subset_1_3)
# print(subset_2_3)

print("Шаг 4\n")

# Теперь потренируемся отбирать нужные нам наблюдения (строки), соответствующие некому условию.
# В dataframe с именем my_stat четыре колонки V1, V2, V3, V4:
# В переменную subset_1 сохраните только те наблюдения, у которых значения переменной V1  строго больше 0,
# и значение переменной V3  равняется 'A'.
# В переменную  subset_2  сохраните только те наблюдения, у которых значения переменной V2  не равняются 10,
# или значения переменной V4 больше или равно 1.
# Как и в предыдущей задаче результат фильтрации - это тоже dataframe.

subset_1_4 = my_stat.query("V1 > 0 & V3 == 'A'")
subset_2_4 = my_stat.query("V2 != 10 | V4 >= 1")
# print(subset_1_4)
# print(subset_2_4)

print("Шаг 5\n")

# Теперь давайте преобразуем наши данные. В переменной my_stat лежат данные с которыми вам необходимо проделать
# следующее дейтвие. В этих данных (my_stat) создайте две новые переменных:
# V5 = V1 + V4
# V6 = натуральный логарифм переменной V2

my_stat["V5"] = my_stat["V1"] + my_stat["V4"]
my_stat["V6"] = np.log(my_stat["V2"])
# print(my_stat.head())

print("Шаг 6\n")

# Отличная работа, закрепим еще пару важных вопросов и можно двигаться дальше.
# Переменные V1, V2  ... такие имена никуда не годятся. С такими названиями легко запутаться в собственных данных
# и в результате ошибиться в расчетах.
# Переименуйте колонки в данных  my_stat следующим образом:
# V1 -> session_value
# V2 -> group
# V3 -> time
# V4 -> n_users

my_stat = pd.read_csv('Lesson 2.8 data/my_stat.csv')
my_stat = my_stat.rename(index=str, columns={"V1": "session_value", "V2": "group", "V3": "time", "V4": "n_users"})
# print(my_stat.head())

print("Шаг 7\n")

# И напоследок давайте разберемся, как заменять наблюдения в данных.
# В dataframe с именем my_stat сохранены данные с 4 колонками: session_value, group, time, n_users.
# В переменной session_value замените все пропущенные значения на нули.
# В переменной n_users замените все отрицательные значения на медианное значение переменной n_users
# (без учета отрицательных значений, разумеется).

my_stat_1 = pd.read_csv("Lesson 2.8 data/my_stat_1.csv")
my_stat_1 = my_stat_1.fillna(0)
filter = my_stat_1["n_users"][my_stat_1["n_users"] < 0].to_list()
median = np.median(my_stat_1["n_users"][my_stat_1['n_users'] >= 0])
my_stat_1['n_users'] = my_stat_1['n_users'].replace(filter, median)
# print(my_stat_1.head(19))

print("Шаг 8\n")

# Чуть не забыли, никакой анализ данных не обходится без агрегации наблюдений.
# Напомню, в pandas с этим нам поможет связка groupby + некоторое преобразование. Например:
# # число наблюдений в каждой группе
# df.groupby('group_var').count()
# Для того, что бы сгруппировать данные по нескольким переменным, используем список с нужными именами:
# df.groupby(['group_var_1', 'group_var_2']).count()
# Обратите внимание, что при такой записи группирующие переменные станут индексами в итоговом dataframe,
# изучите справку по groupby, чтобы разобраться со всеми тонкостями группировки данных.
# Также функция count() применится ко всем колонкам, что не всегда является желанным результатом.
# Чтобы применить функцию только к нужной колонке в данных, можно использовать связку  groupby() + agg()
# my_stat.groupby('group').agg({'n_users': 'count'})
# В этой задаче для данных my_stat рассчитайте среднее значение переменной session_value
# для каждой группы (переменная group), в получившемся dataframe  переменная group не должна превратиться в индекс.
# Также переименуйте колонку со средним значением session_value в mean_session_value.
# Получившийся результат сохраните в dataframe с именем mean_session_value_data.

my_stat_2 = pd.read_csv("Lesson 2.8 data/my_stat_1.csv")
mean_session_value_data = my_stat_2.groupby("group", as_index=False).agg({"session_value": 'mean'})\
    .rename(columns={"session_value": 'mean_session_value'})
# print(mean_session_value_data)
