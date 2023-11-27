import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras

features = pd.read_csv('tmp/temps.csv')
print(features.head())
print(features.shape)

# =========處理時間數據=========
years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + "-" + str(int(month)) + "-" + str(int(day)) 
        for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(dates[:5])

# =========畫圖=========
plt.style.use('fivethirtyeight')
fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
fig.autofmt_xdate(rotation=45)

ax1.plot(dates, features['actual'])
ax1.set_xlabel('')
ax1.set_ylabel('Temperature')
ax1.set_title('Max Temp')

ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('Previous Max Temp')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('Two Days Prior Max Temp')

ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temperature')
ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
plt.show()

# =========獨熱編碼=========
# 原本week 的值換成0,1或True,False
features = pd.get_dummies(features)
print(features.head(5))

# =========建立標籤=========
labels = np.array(features['actual'])
# 在特徵中去掉標籤
features = features.drop('actual', axis=1)
# 保存表頭
features_list = list(features.columns)
# 轉換適合的格式
features = np.array(features)
print(features.shape)

