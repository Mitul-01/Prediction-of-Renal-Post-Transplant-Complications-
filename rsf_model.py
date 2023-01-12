# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:00:14 2022

@author: mitul
"""
from sksurv.metrics import integrated_brier_score
from imblearn.over_sampling import SMOTE
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("kt2_random_data.csv")

print(df.dtypes)

df.isnull().any() #no null values
df.describe()

df_num=df.iloc[:,0:19]
df_cat=df.iloc[:,20:53] 

for i, col in enumerate(df_cat.columns):
    plt.figure(i)
    sns.countplot(df_cat[col])
    
for i, col in enumerate(df_num.columns):
    plt.figure(i)
    sns.distplot(df_num[col])

df1 = df.corr()


########################  SMOTE ###############################################
#################  Synthetic Minority Over-sampling Technique   ###############


y = df.loc[:, [ "Graftsurvivalcensoringindicat"]]
X = df.drop(["Graftsurvivalcensoringindicat"], axis=1)

smote = SMOTE(sampling_strategy = "not majority")

X_smote , y_smote = smote.fit_resample(X,y)


#standarization 
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)

X_norm = norm_func(X_smote)
X_norm.describe()
X_norm.isnull().any()
X_norm = X_norm.fillna(0)

x_norm = X_norm.describe()

final = pd.concat([X_norm,y_smote] , axis = 1)

final

# =============================================================================
# 
# =============================================================================

y1 = final.loc[:, ["Graftsurvivaltime", "Graftsurvivalcensoringindicat"]]
X1 = final.drop(["Graftsurvivalcensoringindicat", "Graftsurvivaltime"], axis=1)

Graftsurvivalcensoringindicat = y1['Graftsurvivalcensoringindicat'].astype(bool)
Graftsurvivaltime = y1["Graftsurvivaltime"]
data  = pd.DataFrame({'Graftsurvivalcensoringindicat':Graftsurvivalcensoringindicat ,'Graftsurvivaltime':Graftsurvivaltime})

y1 = data.to_records(index=False)

X1, X1_test, y1, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=20)

 
# =============================================================================
rsf1 = RandomSurvivalForest(n_estimators=20,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=20)
rsf1.fit(X1, y1)

rsf1.score(X1_test, y1_test) #test #cindex = 0.922287

rsf1.score(X1, y1)#train #cindex = 0.95752

surv_test_rsf = rsf1.predict_survival_function(X1_test, return_array=False)

###############################################################################

lower, upper = np.percentile(y1["Graftsurvivaltime"], [10, 90])
y_times = np.arange(lower, upper+1)

"""
In the below code, I am taking only the range of the Survival 
predicted functions relavent to Random Survival Forest
"""
T1, T2 = surv_test_rsf[0].x.min(),surv_test_rsf[0].x.max()
mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
times = y_times[~mask]

rsf_surv_prob_test = np.row_stack([ fn(times) for fn in surv_test_rsf])
rsf_surv_prob_test


score_brier = pd.Series(
    [
        integrated_brier_score(y1, y1_test, rsf_surv_prob_test, times )
    ])


print(score_brier)


###############################################################################

#Creating a Pickel file

import pickle
pickle_out = open("C:/Users/mitul/Desktop/project work/finalkidneytransplant/rsf1.pkl","wb")
pickle.dump(rsf1,pickle_out)

