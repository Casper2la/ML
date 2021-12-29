import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# All print() is commented out

print("2.10 Stepik ML contest - это еще что такое?\n\n")
print("2.11 Stepik ML contest - data preprocessing\n\n")

print("Шаг 5\n")

# А пока что вот вам хакерская задача, за каким вымышленным id скрывается Анатолий Карпов - автор курса,
# данные которого мы анализируем?
# Введите id Анатолия Карпова, под которым он фигурирует в данных events_data_train и submissions_data_train.

events_data = pd.read_csv('Lessons 1.10-1.11 data/event_data_train.csv')
submissions_data = pd.read_csv('Lessons 1.10-1.11 data/submissions_data_train.csv')

# print(events_data[events_data.action == 'viewed']
#       .groupby('user_id', as_index=False)
#       .agg({'step_id': 'count'})
#       .sort_values(by=['step_id'], ascending=False).head(), '\n')
# print(submissions_data[submissions_data.submission_status == 'correct']
#       .groupby('user_id', as_index=False)
#       .agg({'step_id': 'count'})
#       .sort_values(by=['step_id'], ascending=False).head(), '\n')
#
# print('Верный user_id: 1046', '\n')


