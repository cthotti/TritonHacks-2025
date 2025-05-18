import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from meteostat import Point, Daily, Stations
from datetime import datetime

start = datetime(2024, 5, 3)
#end = datetime(2018, 1, 1)
socal = Point(39.160346,-119.937494)

data = Daily(socal, start,start)
data = data.fetch()

stations = Stations()
stations = stations.nearby(39.160346,-119.937494,radius=10000000)
station = stations.fetch(1)
print(station)

def get_tavg(row):
    try:
        point = Point(row['latitude'], row['longitude'])
        print(type(row['date']))
        date = row['date'].replace(hour=0,minute=0,second=0)
        print(date)
        print(row['latitude'])
        print(row['longitude'])
        daily_data = Daily(point, date, date).fetch()
        print(daily_data)
        if not daily_data.empty and 'tavg' in daily_data.columns:
            return daily_data['tavg'].values[0]
            print('hi')
        else:
            return np.nan
    except Exception as e:
        print(e)
        return None

fire_df = pd.read_csv('mapdataall.csv')
#print(fire_df.columns)
print('###########################################################################################')

fire_df = fire_df[['incident_longitude','incident_latitude','incident_acres_burned','incident_date_created']]
fire_df = fire_df.iloc[:10]
fire_df.columns = ['longitude','latitude','acres_burned','date']
fire_df['date'] = pd.to_datetime(fire_df['date']).dt.tz_localize(None)
fire_df['day_of_year'] = fire_df['date'].dt.dayofyear
fire_df['tavg'] = fire_df.apply(get_tavg,axis=1)
fire_df = fire_df[(fire_df['longitude']<-114) & 
                  (fire_df['latitude']>32) & 
                  (fire_df['latitude']<43)]
fire_df = fire_df.drop(columns=['date'])
#fire_df = fire_df.dropna()

# Optional: Reset index
fire_df = fire_df.reset_index(drop=True)

# 3. Save it back to CSV (overwrite or save as new)
fire_df.to_csv("cleaned_mapdata.csv", index=False)  # index=False avoids writing the index column


large_df = fire_df[fire_df['acres_burned']>1000]
#print(large_df.head)

X = fire_df.drop('acres_burned',axis=1)
y = np.log1p(fire_df['acres_burned'])

"""

to_scale = ['longitude', 'latitude','day_of_year','tavg']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns=to_scale)
print(X)

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

linear_classifier = LinearRegression()
linear_classifier.fit(X_train, y_train)
r2 = r2_score(y_test,linear_classifier.predict(X_test))


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)
print(f'The Random Forest Regressor has a {r2 * 100}% accuracy!')

#y_pred = linear_classifier.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Acres Burned")
plt.show()
"""
"""
############ LATITUDE #################
plt.figure(figsize=(8,6))


weights = [1 / len(large_df)] * len(large_df)

plt.hist(large_df['incident_latitude'], bins=np.arange(32.557546,42.29896,1), weights=weights,
         edgecolor='black', color='skyblue') # Create the histogram

plt.title('large fires')
plt.xlabel('latitude')
plt.ylabel('percentage of fires')
plt.xticks(np.arange(32.557546,42.29896,1))
"""


"""
############ LONGITUDE #################
weights = [1 / len(large_df)] * len(large_df)

plt.hist(large_df['incident_longitude'], bins=np.arange(-125.0,-114.276308,1), weights=weights,
         edgecolor='black', color='skyblue') # Create the histogram

plt.title('large fires')
plt.xlabel('longitude')
plt.ylabel('percentage of fires')
plt.xticks(np.arange(-125.0,-114.276308,1))
"""
"""
plt.gca().set_yticklabels([f'{int(x * 100)}%' for x in plt.gca().get_yticks()]) # Set y-tick labels
plt.show()
"""
"""
longitude max min
-114.276308
-125.0

latitude max min
42.29896
32.557546
"""
