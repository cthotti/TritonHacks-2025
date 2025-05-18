import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from meteostat import Point, Daily, Stations
from datetime import datetime
from dateutil.relativedelta import relativedelta

class FireModel:
    def __init__(self):
        print('hi')
        self.fire_df = pd.read_csv('screen/cleaned_mapdata.csv')
        self.X = self.fire_df.drop('acres_burned',axis=1)
        self.y = np.log1p(self.fire_df['acres_burned'])
        to_scale = ['longitude', 'latitude','day_of_year','tavg','wspd']
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X),columns=to_scale)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=3,random_state=2)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pred_original = np.expm1(y_pred)
        y_test_original = np.expm1(y_test)
        self.MAE = mean_absolute_error(y_test_original,y_pred_original)

    def model_predict(self,latitude,longitude):
        date = datetime.today()
        date = datetime(date.year, date.month, date.day)
        tavg,wspd = self.get_weather(date,latitude,longitude)
        X_input = np.array([[longitude, latitude, date.day, tavg, wspd]])
        columns = ['longitude', 'latitude', 'day_of_year', 'tavg', 'wspd']
        df_input = pd.DataFrame(X_input, columns=columns)
        return self.model.predict(df_input)[0]

    def get_weather(self,date,latitude,longitude):
        date = date - relativedelta(years=5)
        stations = Stations()
        stations = stations.nearby(latitude, longitude, radius=1000000)
        station_df = stations.fetch(10)
        station_df = station_df[station_df['daily_end'] > date]

        if station_df.empty:
            return pd.Series([np.nan, np.nan])

        data = Daily(station_df, date, date)
        data = data.fetch()

        if data.empty:
            return pd.Series([np.nan, np.nan])

        data = data.dropna(subset=['tavg', 'wspd'])

        if not data.empty:
            return pd.Series([data.iloc[0]['tavg'], data.iloc[0]['wspd']])
        else:
            return pd.Series([np.nan, np.nan])

if __name__ == '__main__':
    newModel = FireModel()
    print(f'The XGB Regressor has a {newModel.MAE} accuracy!')
    print(newModel.model_predict(32.160346,-120.937494))
