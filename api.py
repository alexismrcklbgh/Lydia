# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:38:09 2026

@author: AlexisM
"""

from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

rfMajor = joblib.load(BASE_DIR / "rfMajor.joblib")
rfMinor = joblib.load(BASE_DIR / "rfMinor.joblib")
encoder = joblib.load(BASE_DIR / "encoder.joblib")

app = FastAPI(title="Lydia Transaction Classifier")



class Transaction(BaseModel):
    member_id: Optional[int] = None
    amount_value: float
    fees: Optional[int] = None
    created_at: datetime
    merchant_id: Optional[str] = None
    merchant_name: Optional[str] = None
    title: Optional[str] = None
    card_name: Optional[str] = None
    digitalization_type: Optional[str] = None
    merchant_pan_entry_method: Optional[str] = None
    status: Optional[str] = None


def prepare_features(transaction: dict, encoder):
    # Créer une ligne avec les champs de la transaction
    Xtest = pd.DataFrame([transaction])
   
    XtestCat = Xtest.select_dtypes(exclude=[np.number])
    XtestNum = Xtest.select_dtypes(include=[np.number])
    
    mask = XtestCat.notna().all(axis=1)
    XtestCat = XtestCat.loc[mask].reset_index(drop=True)
    XtestNum = XtestNum.loc[mask].reset_index(drop=True)
    
    
    XtestCatEncod = pd.DataFrame(data=encoder.transform(XtestCat))
    
    Xtest = pd.concat([XtestNum, XtestCatEncod], axis=1)
    Xtest.columns = Xtest.columns.astype(str)

    return Xtest




@app.post("/predict")
def predict(payload: Transaction):
    try:
        transaction = payload.model_dump()

        X = prepare_features(transaction, encoder)

        minor_prediction = str(
            rfMinor.predict(X)[0]
        )
        major_prediction = str(
            rfMajor.predict(X)[0]
        )

        return {
            "major_category": major_prediction,
            "minor_category": minor_prediction,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        )
