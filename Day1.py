import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
dataset = pd.read_csv("/home/shimul/program files/CSV/BRRI.csv")
months=dataset.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder
LabelEncoder_months=LabelEncoder()
LabelEncoder_solar_radiation=LabelEncoder()
dataset['Month']=LabelEncoder_months.fit_transform(months)
solar_radiation=dataset.iloc[:,3].values

""" print(f"Max: {dataset['Solar Radiation (w/m^2)'].max()}\n\nMin: {dataset['Solar Radiation (w/m^2)'].min()}\n\nMean: {dataset['Solar Radiation (w/m^2)'].mean()}") """
solar_radiation=solar_radiation>342


dataset["Disease"]=""
dataset['Disease']=LabelEncoder_solar_radiation.fit_transform(solar_radiation)
dataset=dataset.dropna()
dataset=dataset.reset_index(drop=True)

x=dataset.iloc[:,dataset.columns!="Disease"]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor(n_estimators=10)
regressor.fit(x,y)

predictions=regressor.predict(X_test)

print(regressor.score(X_test,y_test))