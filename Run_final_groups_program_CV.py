# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:56:51 2019

@author: D BArrera
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scorecardpy as sc

import seaborn as sns
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation, Reshape

import os



plt.rc("font", size=14)
#------------------------Read Data Base--------------------------
name = 'BBDD.xlsx'
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sheet = "BBDD"
data = pd.read_excel(name, sheet_name=sheet,header=0)
data['ZONE'] = ['UPZ_'+str(i) for i in data['ZONE']]
data = data.dropna()
#-----------------------------Detecting services-------------------------------
SERVICES = np.unique(data['SERVICES'])

data_x = data.copy()

#--------Function to detect Constant Value of Logistic Regresion---------------

def  Logistic_Regression_C(X_train, Y_train, X_test, Y_test):
    #--------------Variation of constant in logistic regression----------------
    Const = np.concatenate([np.arange(0.01,0.09,0.01), np.linspace(0.1,1,10), np.arange(2,11,1)])
    AUC_COEF = pd.DataFrame(index=[Const],columns = ['AUC'])
    COEF = pd.DataFrame({'VARIABLE': X_train.columns})
    #--------Loop to obtain the best possible constant-------------------------
    for k in Const:
        logreg = LogisticRegression(penalty='l1', # Type of penalization l1 = lasso, l2 = ridge
                                             tol=0.0001, # Tolerance for parameters
                                             C=k, # Penalty constant, see below
                                             fit_intercept=True, # Use constant?
                                             class_weight='balanced', # Weights, see below
                                             random_state=20190301, # Random seed
                                             max_iter=100, # Maximum iterations
                                             verbose=1, # Show process. 1 is yes.
                                             solver = 'saga',
                                             warm_start=False # Train anew or start from previous weights. For repeated training.
                                            )
        logreg.fit(X_train,Y_train)
        y_pred_lr_prob = logreg.predict_proba(X_test)
        coef = pd.DataFrame(columns = [str(k)])
        coef[str(k)] =np.transpose(logreg.coef_[0])
        COEF = pd.concat([COEF, coef],
                            axis = 1
                           )
        if np.unique(Y_test).shape[0]>1:
            AUC_COEF['AUC'].loc[k] =  roc_auc_score(Y_test, y_score = y_pred_lr_prob[:,1])
        else:
            AUC_COEF['AUC'].loc[k] = 0


    #Detecting the best AUC to obtain the best constant in LR
    C_best =Const[np.where(AUC_COEF['AUC']==np.max(AUC_COEF['AUC']))[0]][0]
    pos = np.where(COEF[str(C_best)]==0)
    to_erase = COEF['VARIABLE'].loc[pos[0]]
    to_keep = X_train.columns.difference(to_erase)  #Columns to keep
    X_train = X_train[to_keep]
    X_test  = X_test[to_keep]
    COEF_best = COEF[str(C_best)]               
    return C_best, COEF_best, X_train, Y_train, X_test, Y_test, len(pos[0]), AUC_COEF, COEF


#----------Function to generate the dummies variables from categorical variables------------
def dummies_on(train_b, test_b, bins,cat_vars):
    train = train_b.copy()
    test = test_b.copy()
    for i in cat_vars:
        breaks = np.array(bins[i]['breaks'], dtype=float)
        breaks = np.insert(breaks,0,-1)
        category_train = pd.cut(train[i],breaks)
        category_train = category_train.to_frame()
        category_train.columns = ['range']
        train[i]= pd.concat([train[i],category_train],axis = 1)['range']
        category_test = pd.cut(test[i],breaks)
        category_test = category_test.to_frame()
        category_test.columns = ['range']
        test[i]= pd.concat([test[i],category_test],axis = 1)['range']
    cat_vars = np.append(cat_vars,train.columns.difference(cat_vars))
    pos = np.where(cat_vars=='OUTCOME')
    cat_vars = np.delete(cat_vars,pos)
    var_deleted = []
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(train[var], prefix=var)
        var_deleted.append(cat_list.columns[-1])
        cat_list.drop(cat_list.columns[-1],axis = 1, inplace = True)
        data1=train.join(cat_list)
        train=data1
        cat_list = pd.get_dummies(test[var], prefix=var)
        data1=test.join(cat_list)
        test = data1
    data_vars_train=train.columns.values.tolist()
    data_vars_test=test.columns.values.tolist()
    data_vars = list(set(data_vars_train).intersection(data_vars_test))
    to_keep=[i for i in data_vars if i not in cat_vars]
    train=train[to_keep]
    test=test[to_keep]
    return(train,test,var_deleted)

#--------------Prepare excel of the output file--------------------------
name_out = '2017_2018_SERVICES.xls'
random_state_v2 = 123456
continuous = ['AGE','LEAD_TIME'] #Continuous variables

#Looping around the SERVICES
for e in SERVICES:
    #Detecting a particular service
    Full_path = e 
    if not os.path.exists(Full_path):
        os.mkdir(Full_path)
    print(e)
    AUCROC = pd.DataFrame(columns =['LR','RF','NN'], index = range(1,101))
    COEF2 = pd.DataFrame() 
    data = data_x.copy()
    data = data.loc[data['SERVICES'] == e]
    data. drop(['SERVICES'], axis = 1, inplace = True)
#------------------ Grouping zone if there is not enough data------------------------
    Clases_UPZ = np.unique(data['ZONE'])
    Data_UPZ = data.groupby('ZONE').groups
    for i in Clases_UPZ:
        numero_clases = Data_UPZ[i]
        if len(numero_clases)<10:
            data['ZONE'].loc[numero_clases] = 'ZONE_Other'

#-------------------- 1. OBTAINING BINS---------------------------------------
    train_b, test_b = sc.split_df(data, y = 'OUTCOME',
                          ratio = 0.7, seed = 100).values()
    bins = sc.woebin(train_b, y = 'OUTCOME',
                         min_perc_fine_bin=0.01,     # How many bins to cut initially into
                         min_perc_coarse_bin=0.05,   # Minimum percentage per final bin
                         stop_limit=0.2,             # Minimum information value
                         max_num_bin=10,              # Maximum number of bins
                         method='tree')
#---------Define into N splits for CV 10x10
    iteracion = 1
    N = 10 
    kf = RepeatedKFold(n_splits=N, n_repeats=10, random_state=random_state_v2)
    for train_index, test_index in kf.split(data):
        print(iteracion)
        #Divive training and testing set
        train_b, test_b = data.iloc[train_index], data.iloc[test_index]
        #Convert into dummies variables
        train,test,deleted_var = dummies_on(train_b,test_b,bins,continuous)
        #Defining Train Data and Test Data
        X_train = train[train.columns.difference(['OUTCOME'])]
        Y_train = train['OUTCOME']
        Y_train=Y_train.astype('int')
        X_test = test[test.columns.difference(['OUTCOME'])]
        Y_test = np.array(test['OUTCOME'], dtype = float)
        Name_columns = X_test.columns
        Name_columns = Name_columns.union(['number_deleted_variables'])
        COEF = pd.DataFrame(index = Name_columns)
        #Runing LR to obtaing best C and the variables that will be taken into account
        C_best, COEF_best, X_train, Y_train, X_test, Y_test, N_erase, AUC_COEF, COEF_out = Logistic_Regression_C(X_train, Y_train, X_test, Y_test)
        COEF_best.loc[len(COEF_best)] = N_erase
        COEF[str(C_best)] = COEF_best.values
        COEF2 = pd.concat([COEF2, COEF],axis=1, sort = False)
       #----------------------1. LOGISTIC REGRESION---------------------------
        logreg = LogisticRegression(penalty='l1', # Type of penalization l1 = lasso, l2 = ridge
                                    tol=0.0001, # Tolerance for parameters
                                    C=C_best, # Penalty constant, see below
                                    fit_intercept=True, # Use constant?
                                    class_weight='balanced', # Weights, see below
                                    random_state=20190301, # Random seed
                                    max_iter=100, # Maximum iterations
                                    verbose=1, # Show process. 1 is yes.
                                    solver = 'saga',
                                     warm_start=False # Train anew or start from previous weights. For repeated training.
                                    )
    #----------------------2. RANDOM FOREST---------------------------
        #Depending on the SERVICE, parameters can change
        random_forest = RandomForestClassifier(n_estimators=1000, # Number of trees to train
                               criterion='gini', # How to train the trees. Also supports entropy.
                               max_depth=None, # Max depth of the trees. Not necessary to change.
                               min_samples_split=2, # Minimum samples to create a split.
                               min_samples_leaf=0.001, # Minimum samples in a leaf. Accepts fractions for %. This is 0.1% of sample.
                               min_weight_fraction_leaf=0.0, # Same as above, but uses the class weights.
                               max_features=2, # Maximum number of features per split (not tree!) by default is sqrt(vars)
                               max_leaf_nodes=None, # Maximum number of nodes.
                               min_impurity_decrease=0.0001, # Minimum impurity decrease. This is 10^-3.
                               bootstrap=True, # If sample with repetition. For large samples (>100.000) set to false.
                               oob_score=True,  # If report accuracy with non-selected cases.
                               n_jobs=-1, # Parallel processing. Set to -1 for all cores. Watch your RAM!!
                               random_state=20190305, # Seed
                               verbose=1, # If to give info during training. Set to 0 for silent training.
                               warm_start=False, # If train over previously trained tree.
                               class_weight='balanced')
        #----------------------3. NEURAL NETWORK-------------------------------
        # create model
        nn = Sequential()
        dim_in = np.max((1,X_train.shape[1]))
        nn.add(Dense(12, input_dim=dim_in, activation='relu'))      #Depending on the SERVICE, hidden neurons can change
        nn.add(Dense(8, activation='sigmoid'))
        nn.add(Dense(1, activation='sigmoid'))
        # Compile model
        nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
          # Fit the model
        if X_train.shape[1]!=0:
    
             result = logreg.fit(X_train,Y_train)
             y_pred_lr = logreg.predict(X_test)
             y_pred_lr_prob = logreg.predict_proba(X_test)
             if X_train.shape[1]>4:
                 random_forest.fit(X_train,Y_train)
                 y_pred_rf = random_forest.predict(X_test)
                 y_pred_rf_prob = random_forest.predict_proba(X_test)
    
             nn.fit(X_train, Y_train, epochs=1000) #Depending on the SERVICE, epochs can change
             y_pred_nn_prob = nn.predict(X_test)
             if np.unique(Y_test).shape[0]>1:
                AUCROC['LR'].loc[iteracion] = roc_auc_score(Y_test, y_score = y_pred_lr_prob[:,1])
                AUCROC['RF'].loc[iteracion] = roc_auc_score(Y_test, y_score = y_pred_rf_prob[:,1])
                AUCROC['NN'].loc[iteracion] = roc_auc_score(Y_test, y_score = y_pred_nn_prob)
        iteracion+=1            
    #Write result
    writer = pd.ExcelWriter(Full_path + '/' + name_out + '_' + e[0:10] + '.xls', engine='xlsxwriter')
    AUCROC.to_excel(writer,'AUC')
    COEF2.to_excel(writer,'COEF')
    writer.save()





