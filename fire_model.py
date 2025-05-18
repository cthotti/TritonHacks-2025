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
from xgboost import XGBRegressor

def get_weather(row):
    try:
        stations = Stations()
        stations = stations.nearby(row['latitude'], row['longitude'], radius=1000000)
        date = row['date'].replace(hour=0, minute=0, second=0)
        station_df = stations.fetch(10)
        station_df = station_df[station_df['daily_end'] > date]

        if station_df.empty:
            return pd.Series([np.nan, np.nan])

        data = Daily(station_df, date, date)
        data = data.fetch()

        if data.empty:
            return pd.Series([np.nan, np.nan])

        # Drop rows with missing values
        data = data.dropna(subset=['tavg', 'wspd'])

        if not data.empty:
            return pd.Series([data.iloc[0]['tavg'], data.iloc[0]['wspd']])
        else:
            return pd.Series([np.nan, np.nan])

    except Exception as e:
        print(f"Error: {e}")
        return pd.Series([np.nan, np.nan])
fire_df = pd.read_csv('cleaned_mapdata.csv')

"""
fire_df = fire_df[['incident_longitude','incident_latitude','incident_acres_burned','incident_date_created']]
fire_df.columns = ['longitude','latitude','acres_burned','date']
fire_df['date'] = pd.to_datetime(fire_df['date']).dt.tz_localize(None)
fire_df['day_of_year'] = fire_df['date'].dt.dayofyear
fire_df[['tavg', 'wspd']] = fire_df.apply(get_weather, axis=1)
fire_df = fire_df[(fire_df['longitude']<-114) & 
                  (fire_df['latitude']>32) & 
                  (fire_df['latitude']<43)]
fire_df = fire_df.drop(columns=['date'])
fire_df = fire_df.dropna()
fire_df = fire_df[fire_df['tavg']>10]"""

# Optional: Reset index
#fire_df = fire_df.reset_index(drop=True)

# 3. Save it back to CSV (overwrite or save as new)
#fire_df.to_csv("cleaned_mapdata.csv", index=False)  # index=False avoids writing the index column

"""
outlier_upp  = fire_df['acres_burned'].quantile(0.5) + 1.5*(fire_df['acres_burned'].quantile(0.75) - fire_df['acres_burned'].quantile(0.25))
outlier_low  = fire_df['acres_burned'].quantile(0.5) - 1.5*(fire_df['acres_burned'].quantile(0.75) - fire_df['acres_burned'].quantile(0.25))
fire_df = fire_df.sort_values(by='acres_burned',ascending=False)
fire_df = fire_df[fire_df['acres_burned'] < outlier_upp]
fire_df = fire_df[fire_df['acres_burned'] > outlier_low]
fire_df.to_csv("cleaned_mapdata.csv", index=False)
"""
X = fire_df.drop('acres_burned',axis=1)
y = np.log1p(fire_df['acres_burned'])



to_scale = ['longitude', 'latitude','day_of_year','tavg','wspd']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns=to_scale)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


model = RandomForestRegressor(n_estimators=100, max_depth=3,random_state=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)


r2 = mean_absolute_error(y_test_original,y_pred_original)
print(f'The XGB Regressor has a {r2 } accuracy!')


"""#y_pred = linear_classifier.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Acres Burned")
plt.show()"""
