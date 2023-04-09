import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn. compose import ColumnTransformer
from sklearn. preprocessing import LabelEncoder, OrdinalEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, auc
from xgboost.sklearn import XGBClassifier


data = pd.read_csv("survey.csv")
data.drop(['Country', 'state', 'Timestamp', 'comments'], axis = 1, inplace = True)
data['self_employed'].fillna('No', inplace = True)
data['work_interfere'].fillna('N/A', inplace = True)
data.drop(data[(data['Age'] > 80) | (data['Age'] < 18)].index, inplace = True)
data['Gender'].replace(['Male', 'male','M','m','Make','Cis Male','Man','Malr','Cis Man','msle','cis male'
                      'Mail','Guy (-ish) ^_^','Male (CIS)','Male-ish','maile','Mal', 'Mail', 'Male ', 'cis male'], 'Male', inplace = True)
data['Gender'].replace(['Female', 'female','F','f','Woman','Female (cis)','cis-female/femme','femail','Cis Female','Femake',
                        'woman','Female '], 'Female', inplace = True)
data['Gender'].replace(['Female (trans)','Trans woman','male leaning androgynous','Neuter','queer','Guy (-ish) ^_^',
                        'Enby','Agender','Trans-female','something kinda male?','queer/she/they', 'Androgyne', 'non-binary',
                        'Nah','fluid','Genderqueer','ostensibly male, unsure what that really means'], 'Non-binary', inplace = True)
x = data.drop('treatment', axis = 1)
y = data['treatment']
ct = ColumnTransformer([('oe', OrdinalEncoder(),['Gender','self_employed','family_history','work_interfere','no_employees',
                        'remote_work','tech_company','benefits','care_options','wellness_program', 'seek_help', 'anonymity',
                        'leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_health_interview',
                        'phys_health_interview','mental_vs_physical', 'obs_consequence'])], remainder = 'passthrough')
x = ct.fit_transform(x)
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(ct, 'feature_values')
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state = 49)
res = AdaBoostClassifier(random_state=99)
res.fit(x_train,y_train)
pred_res = res.predict(x_test)
print("Accuracy of AdaBoost:", accuracy_score(y_test,pred_res))
from sklearn.model_selection import RandomizedSearchCV
params_res = {'n_estimators': [int(x) for x in np.linspace(start = 1, stop = 50, num = 15)],
              'learning_rate':[(0.97+x/100) for x in range(0,8)]}
res_random = RandomizedSearchCV(random_state = 49,estimator = res,param_distributions = params_res,n_iter=50,cv=5,n_jobs=-1)
res_random.fit(x_train,y_train)
res_tuned = AdaBoostClassifier(random_state = 49,n_estimators=11,learning_rate=1.02)
res_tuned.fit(x_train,y_train)
pred_res_tuned = res_tuned.predict(x_test)