import pandas as pd
import numpy as np
from scipy.stats import entropy

# All print() is commented out

print("3.2 Немного теории и энтропии\n\n")

print("Шаг 7\n")

# В нашем Big Data датасэте появились новые наблюдения! Давайте немного посчитаем энтропию,
# чтобы лучше понять, формализуемость разделения на группы.
# Формат записи - энтропия в группе, где переменная равна 0 и энтропия в группе,
# где переменная равна 1 (десятичный разделитель - точка, округляйте до 2-ого знака при необходимости).

# Энтропия при разделении по фиче Шерстист в группах, где Шерстист равно 0 и 1 соответственно, составляет
# Энтропия при разделении по фиче Гавкает в группах, где Гавкает равно 0 и 1 соответственно, составляет
# Энтропия при разделении по фиче Лазает по деревьям в группах, где эта фича равна 0 и 1 соответственно, составляет

catdog = pd.read_csv('Lesson 3.2 data/cats.csv', index_col=0)
# print(catdog)

feature_fur_0 = [1]
feature_fur_1 = [4/9, 5/9]
feature_bark_0 = [1]
feature_bark_1 = [4/5, 1/5]
feature_climb_0 = [1]
feature_climb_1 = [1]
catdog_entropy = entropy([4/10, 6/10], base=2)

# print('Энтропия при разделении по фиче Шерстист в группах, где Шерстист равно 0:',
#       round(entropy(feature_fur_0, base=2), 2))
# print('Энтропия при разделении по фиче Шерстист в группах, где Шерстист равно 1:',
#       round(entropy(feature_fur_1, base=2), 2))
# print('Энтропия при разделении по фиче Гавкает в группах, где Гавкает равно 0:',
#       round(entropy(feature_bark_0, base=2), 2))
# print('Энтропия при разделении по фиче Гавкает в группах, где Гавкает равно 1:',
#       round(entropy(feature_bark_1, base=2), 2))
# print('Энтропия при разделении по фиче Лазает по деревьям в группах, где эта фича равна 0:',
#       round(entropy(feature_climb_0, base=2), 2))
# print('Энтропия при разделении по фиче Лазает по деревьям в группах, где эта фича равна 1:',
#       round(entropy(feature_climb_1, base=2), 2))

print("Шаг 8\n")

# Ещё немного арифметики - посчитаем Information Gain по данным из предыдущего задания.
# Впишите через пробел округлённые до 2-ого знака значения IG для фичей Шерстист, Гавкает и Лазает по деревьям.
# Десятичным разделителем в данном задании является точка.

IG_fur = round(catdog_entropy -
               (1/10 * entropy(feature_fur_0, base=2) +
                9/10 * entropy(feature_fur_1, base=2)), 2)
IG_bark = round(catdog_entropy -
                (5/10 * entropy(feature_bark_0, base=2) +
                 5/10 * entropy(feature_bark_1, base=2)), 2)
IG_climb = round(catdog_entropy -
                 (4/10 * entropy(feature_climb_0, base=2) +
                  6/10 * entropy(feature_climb_1, base=2)), 2)

# print('IG для фичей Шерстист, Гавкает и Лазает по деревьям:', IG_fur, IG_bark, IG_climb)