import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("2.7 Визуализация, seaborn, почти также круто, как ggplot2\n\n")

df = pd.read_csv('income.csv')
#df_plot = sns.lineplot(data=df) # work
# df_plot2 = df.income.plot() # work
# df_plot3 = df.plot(kind='line') # work
# df_plot4 = df['income'].plot() # work
# df_plot5 = df.plot() # work
# df_plot6 = sns.lineplot(x=df.index, y=df.income) # work
# df_plot7 = plt.plot(df.index, df.income) # work
plt.show()

df = pd.read_csv('dataset_467119_6 (1).txt', delimiter=' ')
print(df.dtypes)
sns.lmplot(x="x", y="y", data=df)
plt.show()