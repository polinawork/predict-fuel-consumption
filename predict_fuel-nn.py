import os
import re
import Prep
import pandas as pd
import numpy as np
from functools import reduce
from math import sqrt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

##folder = "datasets/routes-based dataset/data/NTFWroutes/" #change for datafolder of your choice\
folderLearn = "obra/"
filenames = os.listdir(folderLearn)
pattern = "(route\_\d+\.csv)"
reg = re.compile(pattern)
learnData = [reg.match(z).group(1) for z in filenames if reg.match(z)]
#X = [reg.match(z).group(1) for z in filenames if reg.match(z)]
if not learnData:
    raise ValueError("The folder has no routes available")

folderTest = "test/" #change for datafolder of your choice
filenames = os.listdir(folderTest)
pattern = "(route\_\d+\.csv)"
reg = re.compile(pattern)
testData = [reg.match(z).group(1) for z in filenames if reg.match(z)]
if not testData:
    raise ValueError("The folder has no routes available")


#preparing learning data
ld, lv = [], []
for route in learnData:
    df = pd.read_csv(os.path.join(folderLearn, route))
    data, value= Prep.preproc_fuel2o(df)
    ld.append(data)
    lv.append(value)
ld = reduce(lambda x, y: x.append(y), ld)
lv = reduce(lambda x, y: x.append(y), lv)

#testdata
td1,  tv = [], []
for route in testData:
    df = pd.read_csv(os.path.join(folderTest, route))
    data, value = Prep.preproc_fuel2o(df)
    td1.append(data)
    tv.append(value)
td = reduce(lambda x, y: x.append(y), td1)
tv = reduce(lambda x, y: x.append(y), tv)
tv = np.array(tv)



#predict
scaler = StandardScaler().fit(ld)
standardized_X = scaler.transform(ld)
standardized_X_t = scaler.transform(td)

model = Sequential()
model.add(Dense(10, input_dim = 10, activation = 'relu'))
model.add(Dense (40, activation = 'relu'))
model.add(Dense (1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

model.fit(standardized_X, lv, batch_size = 10, epochs = 100)
y_pr = model.predict(standardized_X_t)

fig = plt.figure(figsize=(10, 5))
plt.title("Приминение нейронных сетей", size=16)
plt.plot(tv,  color='black', label = u'Реальные значения топлива')
plt.plot(y_pr,  color='green', label=u'Предсказанные значения топлива')
plt.xlabel("time", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.legend(fontsize=14)
plt.show()
fig.savefig("NN-obra.png")

#metrics
errors = abs(y_pr - tv)
mape = np.mean(100 * (errors / tv))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

mae = mean_absolute_error(tv, y_pr)
print('MAE: %f' % mae)

mse = mean_squared_error(tv, y_pr)
rmse = sqrt(mse)
print('RMSE: %f' % rmse)

for i in range(len(tv)):
    forecast_errors = tv[i] - y_pr[i]
bias = sum(forecast_errors) * 1.0/len(tv)
print('Bias: %f' % bias)

