import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# ============================
# LOAD DATA
# ============================

# Climate
climate_df = pd.read_csv("data/Climate_data.csv")
# Conflicts
conflicts_df = pd.read_csv("data/East_Africa_conflicts.csv")
# Persons of concern
poc_df = pd.read_csv("data/persons_of_concern.csv")
# Host capacity
capacity_df = pd.read_csv("data/Host_capacity.csv")

# ============================
# FEATURE ENGINEERING
# ============================

# 1. Conflict Count per country-year
conflicts_df['year'] = pd.to_datetime(conflicts_df['WEEK']).dt.year
conflict_summary = conflicts_df.groupby(['COUNTRY', 'year']).size().reset_index(name='Conflict_Count')

# 2. Conflict Intensity: we can simulate average fatalities per event normalized 1-10
conflict_summary2 = conflicts_df.groupby(['COUNTRY', 'year']).agg({'FATALITIES':'sum'}).reset_index()
conflict_summary2['Conflict_Intensity'] = np.clip(conflict_summary2['FATALITIES']/10, 1, 10)

# Merge conflict counts and intensity
conflict_features = pd.merge(conflict_summary, conflict_summary2[['COUNTRY','year','Conflict_Intensity']], 
                             on=['COUNTRY','year'], how='left')

# 3. Climate feature: map weather to numeric
climate_df['Weather'] = climate_df['drought_index']  # Use drought_index as weather numeric

# 4. Persons of concern: total refugees per country-year
poc_summary = poc_df.groupby(['Country of Asylum', 'Year']).agg({'Total':'sum'}).reset_index()
poc_summary.rename(columns={'Country of Asylum':'COUNTRY','Year':'year','Total':'Refugees'}, inplace=True)

# 5. Merge all features
df = pd.merge(conflict_features, climate_df[['country','year','Weather']], left_on=['COUNTRY','year'], 
              right_on=['country','year'], how='left')
df = pd.merge(df, poc_summary, on=['COUNTRY','year'], how='left')

# Fill missing values
df['Conflict_Count'] = df['Conflict_Count'].fillna(0)
df['Conflict_Intensity'] = df['Conflict_Intensity'].fillna(1)
df['Weather'] = df['Weather'].fillna(0.5)
df['Refugees'] = df['Refugees'].fillna(0)

# ============================
# TRAIN MODEL
# ============================

features = ['Conflict_Count','Conflict_Intensity','Weather','year']
X = df[features]
y = df['Refugees']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.joblib")
print("✅ Model trained and saved as model.joblib")
