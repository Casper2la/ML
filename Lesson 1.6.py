import pandas as pd
# All print() is commented out
print("2.6 Группировка и агрегация, ничего, скоро привыкнем\n\n")
# Пересчитаем число ног у героев игры Dota2! Сгруппируйте героев из датасэта по числу их ног (колонка legs),
# и заполните их число в задании ниже.
# Данные взяты отсюда, на этом же сайте можно найти больше разнообразных данных по Dota2.

# Заполните пропуски:
#     Число героев с "0 ног" -
#     Число героев с 2-мя ногами -
#     Число героев с 4-мя ногами -
#     Число героев с 6-ю ногами -
#     Число героев с 8-ю ногами -


print("Шаг 5")

dotaset = pd.read_csv("Lesson 1.6 data/dota_hero_stats.csv")

legs = dotaset.groupby(["legs"])[["name"]].count()
# print(legs)

# К нам поступили данные из бухгалтерии о заработках Лупы и Пупы за разные задачи! Посмотрите у кого из них больше
# средний заработок в различных категориях (колонка Type) и заполните таблицу, указывая исполнителя с большим заработком
# в каждой из категорий.

print("Шаг 6")

pl_pay = pd.read_csv("Lesson 1.6 data/accountancy.csv")
mean_pay = pl_pay.groupby(["Type", "Executor"]).aggregate({"Salary": 'mean'})
# print(mean_pay)

# Продолжим исследование героев Dota2. Сгруппируйте по колонкам attack_type и primary_attr и
# выберите самый распространённый набор характеристик.

print("Шаг 7")

character = dotaset.groupby(["attack_type", "primary_attr"]).aggregate({"attack_type": 'count',
                                                                        "primary_attr": 'count'})
# print(character)

# Аспирант Ростислав изучает метаболом водорослей и получил такую табличку. В ней он записал вид каждой водоросли,
# её род (группа, объединяющая близкие виды), группа (ещё одно объединение водорослей в крупные фракции) и концентрации
# анализируемых веществ.
# Помогите Ростиславу найти среднюю концентрацию каждого из веществ в каждом из родов (колонка genus)!
# Для этого проведите группировку датафрэйма, сохранённого в переменной concentrations, и примените метод,
# сохранив результат в переменной mean_concentrations.

print("Шаг 8")

concentrations = pd.read_csv("Lesson 1.6 data/algae.csv")
mean_concentrations = concentrations.groupby(["genus"]).aggregate({"sucrose": "mean", "alanin": "mean",
                                                                   'citrate': "mean", 'glucose': "mean",
                                                                   'oleic_acid': "mean"})
# print(mean_concentrations)

# Пользуясь предыдущими данными, укажите через пробел (без запятых) чему равны минимальная,
# средняя и максимальная концентрации аланина (alanin) среди видов рода Fucus.
# Округлите до 2-ого знака, десятичным разделителем является точка.
#
# Формат ответа:
# 0.55 6.77 7.48

print("Шаг 9")

alanin = concentrations[concentrations["genus"] == "Fucus"]

# print("min mean max", round(alanin["alanin"].min(), 2), round(alanin["alanin"].mean(), 2),
#      round(alanin["alanin"].max(), 2))

# Сгруппируйте данные по переменной group и соотнесите вопросы с ответами
print("Шаг 10")

conc_count = concentrations.groupby(["group"])[["species"]].count()
conc_glucose = concentrations.groupby(["group"])["glucose"].apply(lambda x: x.astype(float).max() - x.min())
conc_var = concentrations.groupby(["group"])[["glucose"]].var()
# print(concentrations.head())
# print(conc_count, "\n", conc_glucose, "\n", conc_var)
