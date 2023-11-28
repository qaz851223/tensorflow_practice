import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing

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
# plt.style.use('fivethirtyeight')
# fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
# fig.autofmt_xdate(rotation=45)

# ax1.plot(dates, features['actual'])
# ax1.set_xlabel('')
# ax1.set_ylabel('Temperature')
# ax1.set_title('Max Temp')

# ax2.plot(dates, features['temp_1'])
# ax2.set_xlabel('')
# ax2.set_ylabel('Temperature')
# ax2.set_title('Previous Max Temp')

# ax3.plot(dates, features['temp_2'])
# ax3.set_xlabel('Date')
# ax3.set_ylabel('Temperature')
# ax3.set_title('Two Days Prior Max Temp')

# ax4.plot(dates, features['friend'])
# ax4.set_xlabel('Date')
# ax4.set_ylabel('Temperature')
# ax4.set_title('Friend Estimate')

# plt.tight_layout(pad=2)
# plt.show()

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


# =========轉換數據(預處理)=========
input_features = preprocessing.StandardScaler().fit_transform(features)
print(input_features[0]) 

# =========建構模型=========
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))

model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=100, batch_size=64)
model.summary()

predict = model.predict(input_features)
print(predict.shape)

# =========畫圖=========
true_data = pd.DataFrame(data={'date':dates, 'actual':labels})

years = features[:, features_list.index('year')]
months = features[:, features_list.index('month')]
days = features[:, features_list.index('day')]

test_datas = [str(int(year)) + "-" + str(int(month)) + "-" + str(int(day)) 
        for year, month, day in zip(years, months, days)]
test_datas = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_datas]

predictions_data = pd.DataFrame(data={'date': test_datas, 'prediction': predict.reshape(-1)})

# 真實值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 預測值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()