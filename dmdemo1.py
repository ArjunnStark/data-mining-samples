import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df= pd.DataFrame({
    'age': [19,20,18,np.nan,40],
    'salary':[1000,60000,5500,np.nan,5000],
    'score' :[100,40,60,np.nan,20]
})
print("orignal data")
print(df)
df['age'].fillna(df['age'].mean(),inplace=True)
df['salary'].fillna(df['salary'].median(),inplace=True)
df['score'].fillna(df['score'].mode()[0],inplace=True)
print("after handling missing values")
print(df)
scaler=MinMaxScaler()
normal=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
print("normalized data")
print(normal)