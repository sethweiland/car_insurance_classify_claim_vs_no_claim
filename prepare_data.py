import pandas as pd
import numpy as np


def prep_for_modeling(df):
    ids = df["ID"].values.copy()
    df.drop("ID", axis=1,inplace=True)
    X = df.drop(["CLAIM_FLAG", "CLM_AMT", "CLM_FREQ", 
               "REVOKED"],axis=1).copy()
    return X
