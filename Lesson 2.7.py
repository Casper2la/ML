import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# All print() is commented out
print("2.7 Визуализация, seaborn, почти также круто, как ggplot2\n\n")

print("Шаг 5\n")

# Представьте, что у вас есть датафрэйм df, хранящий данные о зарплате за месяц, со всего 1-ой колонкой income.
# Укажите верные способы, как отрисовать простой график зависимости зарплаты от даты (то, как отображается дата
# сейчас не важно, главное сам график)
# Убедитесь, что вы используте версию seaborn > = 0.9.

df = pd.read_csv('Lesson 2.7 data/income.csv')
# df_plot = sns.lineplot(data=df) # work
# df_plot2 = df.income.plot() # work
# df_plot3 = df.plot(kind='line') # work
# df_plot4 = df['income'].plot() # work
# df_plot5 = df.plot() # work
# df_plot6 = sns.lineplot(x=df.index, y=df.income) # work
# df_plot7 = plt.plot(df.index, df.income) # work
# plt.show()

print("Шаг 6\n")

# Вам дан датасэт с 2-мя фичами (колонками). Постройте график распределения точек (наблюдений) в пространстве
# этих 2-ух переменных (одна из них будет x, а другая - y) и напишите число кластеров, формируемых наблюдениями.
# В ответе вы должны указать число кластеров в виде числа (например: 3).

claster = pd.read_csv('Lesson 2.7 data/dataset_467119_6 (1).txt', delimiter=' ')

# sns.lmplot(x="x", y="y", data=claster)
# plt.show()

print("Шаг 7\n")

# Скачайте данные, представляющие геномные расстояния между видами, и постройте тепловую карту, чтобы различия
# было видно наглядно. В ответ впишите, какая картинка соответствует скачанным данным.
# Чтобы график отображался как на картинках, добавьте
# g = # ваш код для создания теплокарты, укажите параметр cmap="viridis" для той же цветовой схемы
# g.xaxis.set_ticks_position('top')
# g.xaxis.set_tick_params(rotation=90)

genome = pd.read_csv("Lesson 2.7 data/genome_matrix.csv", index_col=0)

# g = sns.heatmap(genome, cmap="viridis", square=True, linewidths=.5, cbar_kws={"shrink": .5})
# g.xaxis.set_ticks_position('top')
# g.xaxis.set_tick_params(rotation=90)
# plt.show()

print("Шаг 8\n")

# Пришло время узнать, кто самый главный рак какая роль в dota самая распространённая.
# Скачайте датасэт с данными о героях из игры dota 2 и посмотрите на распределение их возможных ролей
# в игре (колонка roles). Постройте гистограмму, отражающую скольким героям сколько ролей приписывается
# (по мнению Valve, конечно) и напишите какое число ролей у большинства героев.
# Это задание можно выполнить многими путями, и рисовать гистограмму вообще говоря для этого не нужно.

dota_roles = pd.read_csv("Lesson 2.7 data/dota_hero_stats.csv", index_col=0)

dota_roles["roles_count"] = dota_roles["roles"].str.count(',')+1
# dota_roles.roles_count.hist()
# print(dota_roles['roles_count'].mode())
# plt.show()

print("Шаг 9\n")

# Теперь перейдём к цветочкам. Магистрантка Адель решила изучить какие бывают ирисы.
# Помогите Адель узнать об ирисах больше - скачайте датасэт со значениями параметров ирисов,
# постройте их распределения и отметьте правильные утверждения, глядя на график.
# Распределение должно быть по всем образцам, без разделения на вид.
# Чтобы построить на 1-ом графике распределения для каждого из параметров, можно воспользоваться петлёй
# for column in df:
#     # Draw distribution with that column

iris = pd.read_csv("Lesson 2.7 data/iris.csv", index_col=0)
# sns.kdeplot(data=iris)
# plt.show()

print("Шаг 10\n")

# Рассмотрим длину лепестков (petal length) подробнее и воспользуемся для этого violin плотом.
# Нарисуйте распределение длины лепестков ирисов из предыдущего датасэта с помощью violin плота
# и выберите правильный (такой же) вариант среди предложенных

# sns.violinplot(y=iris["petal length"])
# plt.show()

print("Шаг 11\n")

# Продолжаем изучение ирисов! Ещё один важный тип графиков - pairplot,
# отражающий зависимость пар переменных друг от друга, а также распределение каждой из переменных.
# Постройте его и посмотрите на scatter плоты для каждой из пар фичей. Какая из пар навскидку имеет наибольшую корреляцию?
# Также обратите внимание, что можно разделить на группы с помощью параметра hue.

# sns.pairplot(iris, hue="species")
# plt.show()
