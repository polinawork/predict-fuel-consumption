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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

##folder = "datasets/routes-based dataset/data/NTFWroutes/" #change for datafolder of your choice\
folderLearn = "obra/"
filenames = os.listdir(folderLearn)
pattern = "(route\_\d+\.csv)"
reg = re.compile(pattern)
learnData = [reg.match(z).group(1) for z in filenames if reg.match(z)]
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
    data, value = Prep.preproc_fuel1(df)
    ld.append(data)
    lv.append(value)
ld = reduce(lambda x, y: x.append(y), ld)
lv = reduce(lambda x, y: x.append(y), lv)

#testdata
td, tv = [], []
for route in testData:
    df = pd.read_csv(os.path.join(folderTest, route))
    data, value = Prep.preproc_fuel1(df)
    td.append(data)
    tv.append(value)
td = reduce(lambda x, y: x.append(y), td)
tv = reduce(lambda x, y: x.append(y), tv)
tv = np.array(tv)

#predict
scaler = StandardScaler().fit(ld)
standardized_X = scaler.transform(ld)
standardized_X_t = scaler.transform(td)

#Choose n_features
#estimators = np.arange(5, 28, 1)
#accuracy = []
#for n in estimators:
#    rf = RFE(RandomForestRegressor(n_estimators= 500), n)
#    rf.fit(standardized_X, lv)
#    y_pr = rf.predict(standardized_X_t)
#    tv = np.array(tv)
#    errors = abs(y_pr - tv)
#    mape = np.mean(100 * (errors / tv))
#    accuracy.append(100 - mape)
#    print(n, ' Accuracy:', round((100 - mape), 2), '%.')
#fig = plt.figure(figsize=(10, 5))
#plt.title("Выбор параметра n_features", size=16)
#plt.xlabel("n_features", fontsize=14)
#plt.ylabel("accuracy", fontsize=14)
#plt.plot(estimators, accuracy, 'r')
#fig.savefig("final_choose-n_features.png")

rf = RFE(RandomForestRegressor(n_estimators = 1000, max_features= 8), 10)
fit = rf.fit(standardized_X, lv)
y_pr = rf.predict(standardized_X_t)

fig = plt.figure(figsize=(10, 5))
plt.title("Приминение алгоритма Random Forest", size=16)
plt.plot(tv, 'r', color='black', label = u'Реальные значения топлива')
plt.plot(y_pr, 'r', color='green', label=u'Предсказанные значения топлива')
plt.xlabel("time", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.legend(fontsize=14)
plt.show()
#fig.savefig("RF-obra.png")

#Metrics
errors = abs(y_pr - tv)
mape = np.mean(100 * (errors / tv))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

y_pr = np.array(y_pr)
mae = mean_absolute_error(tv, y_pr)
print('MAE: %f' % mae)

mse = mean_squared_error(tv, y_pr)
rmse = sqrt(mse)
print('RMSE: %f' % rmse)


for i in range(len(tv)):
    forecast_errors = tv[i]- y_pr[i]
bias = forecast_errors * 1.0/len(tv)
print('Bias: %f' % bias)
