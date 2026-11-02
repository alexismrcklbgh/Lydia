# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:07:32 2026

@author: AlexisM
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path



def train_random_forest(X, Y, n_trees=300):
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=10,
        min_samples_split=2,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X, Y)
    return clf


BASE_DIR = Path(__file__).resolve().parent

dataset = pd.read_csv(BASE_DIR / "transaction-dataset.csv")

X = dataset.iloc[:, : -2]
Y = dataset.iloc[:,  -2 :]


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=41)


#################################################### TRAINING  MAJOR ##################################################################


# for i in range(Xtrain.shape[1]):
#     print(str(Xtrain.columns[i]) + ": " + str(Xtrain[Xtrain.columns[i]].nunique()) + " unique values")

Xtrain = Xtrain.drop(["pfm_recurring_frequency", "amount_currency", "card_type", "transaction_id", "ref_id", "member_name"], axis=1).reset_index(drop=True)
Ytrain = Ytrain.reset_index(drop=True)


XtrainCat = Xtrain.select_dtypes(exclude=[np.number])
XtrainNum = Xtrain.select_dtypes(include=[np.number])


XtrainCat.isna().sum() #merchant_name = 96
XtrainNum.isna().sum()

Ytrain.isna().sum()
Ytrain["major_category"] = Ytrain["major_category"].fillna("refund")

mask = XtrainCat.notna().all(axis=1)
XtrainCat = XtrainCat.loc[mask].reset_index(drop=True)
XtrainNum = XtrainNum.loc[mask].reset_index(drop=True)
Ytrain = Ytrain.loc[mask].reset_index(drop=True)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
XtrainCatEncod = pd.DataFrame(data=encoder.fit_transform(XtrainCat))

Xtrain = pd.concat([XtrainNum, XtrainCatEncod], axis=1)
Xtrain.columns = Xtrain.columns.astype(str)

rfMajor = train_random_forest(Xtrain, Ytrain['major_category'])


##################################### TEST  MAJOR #############################################################


Xtest = Xtest.drop(["pfm_recurring_frequency", "amount_currency", "card_type", "transaction_id", "ref_id", "member_name"], axis=1).reset_index(drop=True)
Ytest= Ytest.reset_index(drop=True)

XtestCat = Xtest.select_dtypes(exclude=[np.number])
XtestNum = Xtest.select_dtypes(include=[np.number])


XtestCat.isna().sum() #merchant_name = 17
XtestNum.isna().sum()

Ytest.isna().sum()
Ytest["major_category"] = Ytest["major_category"].fillna("refund")

mask = XtestCat.notna().all(axis=1)
XtestCat = XtestCat.loc[mask].reset_index(drop=True)
XtestNum = XtestNum.loc[mask].reset_index(drop=True)
Ytest = Ytest.loc[mask].reset_index(drop=True)


XtestCatEncod = pd.DataFrame(data=encoder.transform(XtestCat))

Xtest = pd.concat([XtestNum, XtestCatEncod], axis=1)
Xtest.columns = Xtest.columns.astype(str)


pred = rfMajor.predict(Xtest)

acc = accuracy_score(Ytest['major_category'],pred)
prec = precision_score(Ytest['major_category'], pred, average=None)
rec = recall_score(Ytest['major_category'], pred, average=None)

f1 = f1_score(Ytest['major_category'], pred, average=None)

print(classification_report(
    Ytest['major_category'],
    pred,
    zero_division=0
))


#################################################### TRAINING  MINOR ##################################################################




rfMinor = train_random_forest(Xtrain, Ytrain['minor_category'])


##################################### TEST  MINOR #############################################################


predMinor = rfMinor.predict(Xtest)

accMinor = accuracy_score(Ytest['minor_category'],predMinor)
precMinor = precision_score(Ytest['minor_category'], predMinor, average=None)
recMinor = recall_score(Ytest['minor_category'], predMinor, average=None)

f1Minor = f1_score(Ytest['minor_category'], predMinor, average=None)

print(classification_report(
    Ytest['minor_category'],
    predMinor,
    zero_division=0
))

########################################## SAVE MODELS ##################################################################

joblib.dump(rfMajor, r"C:\Users\AlexisM\Desktop\Lydia\rfMajor.joblib")
joblib.dump(rfMinor, r"C:\Users\AlexisM\Desktop\Lydia\rfMinor.joblib")

joblib.dump(encoder, r"C:\Users\AlexisM\Desktop\Lydia\encoder.joblib")
































