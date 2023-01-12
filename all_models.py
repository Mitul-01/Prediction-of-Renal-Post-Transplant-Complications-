# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 17:59:33 2022

@author: mitul
"""

from pandas_profiling import ProfileReport
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sksurv.metrics import integrated_brier_score
from imblearn.over_sampling import SMOTE
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

df = pd.read_csv("kt2_random_data.csv")

print(df.dtypes)

df.isnull().any() #no null values
df1 = df.describe()

df_num=df.iloc[:,0:19]
df_cat=df.iloc[:,20:53] 

for i, col in enumerate(df_cat.columns):
    plt.figure(i)
    sns.countplot(df_cat[col])
    
for i, col in enumerate(df_num.columns):
    plt.figure(i)
    sns.distplot(df_num[col])

df1 = df.corr()

########################## Auto EDA ###########################################
profile = ProfileReport(df)
profile.to_file(output_file = "kidneytransplant.html")

########################  SMOTE ###############################################
#################  Synthetic Minority Over-sampling Technique   ###############


y = df.loc[:, [ "Graftsurvivalcensoringindicat"]]
X = df.drop(["Graftsurvivalcensoringindicat"], axis=1)

smote = SMOTE(sampling_strategy = "not majority")

X_smote , y_smote = smote.fit_resample(X,y)


#standarization   #mean=0 , std=1
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)

X_norm = norm_func(X_smote)
X_norm.describe()
X_norm.isnull().any()
X_norm = X_norm.fillna(0)


final = pd.concat([X_norm,y_smote] , axis = 1)

final

##############   Randomsurvivalforest using sksurv ############################

y1 = final.loc[:, ["Graftsurvivaltime", "Graftsurvivalcensoringindicat"]]
X1 = final.drop(["Graftsurvivalcensoringindicat", "Graftsurvivaltime"], axis=1)

Graftsurvivalcensoringindicat = y1['Graftsurvivalcensoringindicat'].astype(bool)
Graftsurvivaltime = y1["Graftsurvivaltime"]
data  = pd.DataFrame({'Graftsurvivalcensoringindicat':Graftsurvivalcensoringindicat ,'Graftsurvivaltime':Graftsurvivaltime})

y1 = data.to_records(index=False) #converting to dataframe to structured array

X1, X1_test, y1, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=20)

rsf = RandomSurvivalForest(n_estimators=20,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=20)
rsf.fit(X1, y1)

rsf.score(X1_test, y1_test) #test #cindex = 0.922287

rsf.score(X1, y1)#train #cindex = 0.95752


######################  COXNET  ##############################################

cph = CoxnetSurvivalAnalysis(l1_ratio=0.06, fit_baseline_model=True)
cph.fit(X1,y1)
cph.score(X1_test, y1_test) #test #cindex = 0.955025  #good model
cph.score(X1, y1)#train #cindex = 0.9786225  #good model
    
####################### STANDARD COX ##########################################

cox1 = CoxPHSurvivalAnalysis(alpha=0.01)
cox1.fit(X1, y1)

cox1.score(X1_test, y1_test) #test #cindex = 0.9927

cox1.score(X1, y1)#train  #cindex = 0.99


###############################################################################

"""
We first need to determine for which time points t,
we want to compute the Brier score for.
We are going to use a data-driven approach here by selecting all 
time points between the 10% and 90% percentile of observed time points.
"""
lower, upper = np.percentile(y1["Graftsurvivaltime"], [10, 90])
y_times = np.arange(lower, upper + 1)


############################################################################
surv_test_rsf = rsf.predict_survival_function(X1_test, return_array=False)

"""
In the below code, I am taking only the range of the Survival 
predicted functions relavent to Random Survival Forest
"""
T1, T2 = surv_test_rsf[0].x.min(),surv_test_rsf[0].x.max()
mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
times = y_times[~mask]


rsf_surv_prob_test = np.row_stack([ fn(times) for fn in surv_test_rsf  ])
rsf_surv_prob_test
"""

These arrays are likely predictions of the survival time of a kidney 
transplant patient in days, months, and years, respectively.

"""
score_brier_test = pd.Series(
    [
        integrated_brier_score(y1, y1_test, rsf_surv_prob_test, times )
    ])

print(score_brier_test)


###############################################################################

surv_test_cph = cph.predict_survival_function(X1_test)
T1, T2 = surv_test_cph[0].x.min(),surv_test_cph[0].x.max()
mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
times = y_times[~mask]

cph_surv_prob_test = np.row_stack([ fn(times) for fn in surv_test_cph  ])
cph_surv_prob_test

cphscore_brier_test = pd.Series(
    [
        integrated_brier_score(y1, y1_test, cph_surv_prob_test, times )
    ])

print(cphscore_brier_test)



##############################################################################

surv_test_cox1 = cox1.predict_survival_function(X1_test)


T1, T2 = surv_test_cox1[0].x.min(),surv_test_cox1[0].x.max()
mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
times = y_times[~mask]

cox1_surv_prob_test = np.row_stack([ fn(times) for fn in surv_test_cox1 ])
cox1_surv_prob_test

coxscore_brier_test = pd.Series(
    [
        integrated_brier_score(y1, y1_test, cox1_surv_prob_test, times)
    ])

print(coxscore_brier_test)




