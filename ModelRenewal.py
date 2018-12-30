#Присоединяем библиотеки для работы расчетов - pandas, numpy, matplotlib
#Import libraries to porogram pandas, numpym matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
print(getcwd())

#Открываем файл csv содержащий данные программы, импортируем данные в датафрейм pandas
#Open csv file with data, import data to pandas dataframe

# перезапись файла из сети
url = 'https://github.com/MindSetLib/PythonUsefull/raw/master/BinaryClassificationInsuranceRenewal/InsuranceRenewalDB.csv'
data = pd.read_csv(url, sep=';', encoding='utf-8',nrows=3000)

'''
fileW=open('resourses\\InsuranceRenewalDB.csv','w')
dataW.to_csv(fileW)

f = open('resourses\\InsuranceRenewalDB.csv')
data = pd.read_csv(f, sep=';', encoding='utf-8')
'''
print (list(data.columns))

data=data[[
    'DATA_TYPE',
    'POLICY_ID',
    'POLICY_BEGIN_MONTH',
    'POLICY_END_MONTH',
    'POLICY_IS_RENEWED',
    'POLICY_SALES_CHANNEL',
    'POLICY_SALES_CHANNEL_GROUP',
    'POLICY_BRANCH',
    'POLICY_MIN_AGE',
    'POLICY_MIN_DRIVING_EXPERIENCE',
    'VEHICLE_MAKE',
    'VEHICLE_MODEL',
    'VEHICLE_ENGINE_POWER',
    'VEHICLE_IN_CREDIT',
    'VEHICLE_SUM_INSURED',
    'POLICY_INTERMEDIARY',
    'INSURER_GENDER',
    'POLICY_CLM_N',
    'POLICY_CLM_GLT_N',
    'POLICY_PRV_CLM_N',
    'POLICY_PRV_CLM_GLT_N',
    'CLIENT_HAS_DAGO',
    'CLIENT_HAS_OSAGO',
    'POLICY_COURT_SIGN',
    'CLAIM_AVG_ACC_ST_PRD',
    'POLICY_HAS_COMPLAINTS',
    'POLICY_YEARS_RENEWED_N',
    'POLICY_DEDUCT_VALUE',
    'CLIENT_REGISTRATION_REGION',
    'POLICY_PRICE_CHANGE'

]]

#Исследуем данные - посмотрим на верхние 5 строк
#Explortory data analysis - have a loot at the data
data.head()


# Посмотрим на размер датафрейма
# Let's have a loot at dataframe shape
data.shape

# Детальнее взглянем на данные и ключевые статистики по ним
# Lets have a look at details and key statistics
data.describe()



#В датафрейме есть два типа данных тестовые и тренировочные. Возмем тренировчные данные, тестовые данные не содержат целевой перменной
#Thare are two types of data - test and train. Lets take train dataset
data_train = data.loc[ data['DATA_TYPE'] == 'TRAIN' ]
data_test = data.loc[ data['DATA_TYPE'] == 'TEST ' ]


#Зададим целевую переменную POLICY_IS_RENEWED - полис был пролонгирован
#Set target variable - Policy was renewed
Y = data_train['POLICY_IS_RENEWED']


#Удалим ID полиса из обучающей выборки
#Remove policy ID from train data
data_train=data_train.drop(('POLICY_ID'), axis=1)


#Удалим целевую переменную из обучающей выборки
#Remove target variable from train data
data_train=data_train.drop(('POLICY_IS_RENEWED'), axis=1)


#Удалим тип данных из обучающей выборки
#Remove data type from train data
data_train=data_train.drop(('DATA_TYPE'), axis=1)



#Преобразуем признак изменения цены полиса в числовую переменную
#Transform POLICY_PRICE_CHANGE to numeric variable
data_train.POLICY_PRICE_CHANGE = pd.to_numeric(data_train.POLICY_PRICE_CHANGE)


#Определим по типу данных категориальные признаки и числовые для дальнейших преобразований
#Find categorical and numerical columns for further transformation
categorical_columns = [c for c in data_train.columns if data_train[c].dtype.name == 'object']
numerical_columns   = [c for c in data_train.columns if data_train[c].dtype.name != 'object']



# Посмотрим на категориальные признаки
# Have a look at categorical variables
data_train[categorical_columns].describe()

#Заполним пропущенные значения категориальных признаков самыми популярными значениеми
#Fill fields with not available date by top values
data_describe = data_train.describe(include=[object])
for c in categorical_columns:
    data_train[c] = data_train[c].fillna(data_describe[c]['top'])



# Заполним пропущенные значения числовых признаков медианным значанием
# Fill numerical data train values by median values
data_train = data_train.fillna(data.median(axis=0), axis=0)


# Посмотрим на корреляцию между данными
# Look at variables correlation
data_train.corr()


# Определим переменные с количеством категорий  2 (бинарные) и более двух (многоклассовые)
# Find variables with 2 categories and more than 2
binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print ("Binary: ", binary_columns)
print("Nonbinary: ", nonbinary_columns)



# Заменим значения бинарных категорий признаков на значения 0 или 1
# Change binary categroies to 0 or 1
for c in binary_columns[0:]:
    top = data_describe[c]['top']
    top_items = data_train[c] == top
    data_train.loc[top_items, c] = 0
    data_train.loc[np.logical_not(top_items), c] = 1


# Создадим новый датафрейм с категориальными признаками преобразовынными в dummy переменные 1 и 0 вместо значения каждой категории
# Create new dataframe with categorical variables transformed to dummy variable instead
data_nonbinary = pd.get_dummies(data_train[nonbinary_columns])
print (data_nonbinary.columns)



# Ряд алгоритмов требует нормализованного пространства признаков. Нормализуем пространство
# Some of ML algorithms need to normalise dataset
data_numerical = data_train[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
data_numerical.describe()



# Объедним массивы данных и преоброзуем все значения к типу float, количество признаков вырасло с 28 до 2705
# Concatenate data arrays and transform all values to float, number of variables increase from 28 to 2705
data_model = pd.concat((data_numerical, data_train[binary_columns], data_nonbinary), axis=1)
data_model = pd.DataFrame(data_model, dtype=float)
print ("Shape: ",data_model.shape)
print ("Columns: ",data_model.columns)



# Назовем входные параметры для модели X
# Lets name X input parameters
X = data_model.copy()
feature_names = X.columns




#Импортируем стандартный набор компопнентов бибилотеки sklearn
#Import components of sklearn library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Задаем печать различные метрики качества модели
# Def different quality metrics
def MetricsPrint(model):
    print("precision:",metrics.precision_score(y_test, model.predict(X_test)))
    print("recall:",metrics.recall_score(y_test, model.predict(X_test)))
    print("roc_auc:",roc_auc_score(y_test, model.predict(X_test)))
    print("gini:",2*roc_auc_score(y_test, model.predict(X_test))-1)
    print ("accuracy:",accuracy_score(y_test, model.predict(X_test)))

# Поделим выборку для модели на 2 части - тестовую (30 процентов) - измерение качества и тренировочную (70 процентов) - построение модели.
# Split data to 2 parts - test - measure of quality and train - model building
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 11)


# Построим модель случайного леса, с числом деревьев 100 и фиксированным параметром случайности
# Build model of random forest with number of trees 100 and fixed random state = 11
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

MetricsPrint(rf)



# Сортируем метрики качества по степени их убывания
# Sort metrics by metrics importance

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))



# Построим альтернативную модель с помощью градиентного бустинга над деревьями с тем же количеством деревьев
# Build alternative model with trees gradient boosting with the same number of trees

from sklearn import ensemble
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train, y_train)


# Рассчитаем различные метрики качества модели
# Calculate different quality metrics of gradient boosting model

MetricsPrint(gbt)

# linear models

from sklearn import linear_model
lr = linear_model.LogisticRegression(random_state=11)
lr.fit(X_train, y_train)


# Рассчитаем различные метрики качества модели
# Calculate different quality metrics of gradient boosting model

MetricsPrint(lr)


import pickle
# Сериализация и десиарилизация pickle
with open('serialisation\\RF_Model.pickle', 'wb') as f:
    pickle.dump(rf, f)

with open('serialisation\\LR_Model.pickle', 'wb') as f:
    pickle.dump(lr, f)

with open('serialisation\\GBT_Model.pickle', 'wb') as f:
    pickle.dump(gbt, f)

#--------------------------------
# десериализация

with open('serialisation\\RF_Model.pickle', 'rb') as f:
    RF_Model = pickle.load(f)

with open('serialisation\\LR_Model.pickle', 'rb') as f:
    LR_Model = pickle.load(f)

with open('serialisation\\GBT_Model.pickle', 'rb') as f:
    GBT_Model = pickle.load(f)


# Сериализация и десиарелизация pmml

