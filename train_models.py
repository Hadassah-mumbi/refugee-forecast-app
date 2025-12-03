"""
train_models.py - ML Model Training using REAL refugee datasets
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# JSON converter for numpy types
# -----------------------------
def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# -----------------------------
# Load and merge datasets
# -----------------------------
def load_and_merge_datasets():
    """Load and merge all 4 CSV files"""
    print("📂 Loading real refugee datasets...")
    
    try:
        # 1. Load Demographics Data
        print("   Loading demographics data...")
        demo_df = pd.read_csv('data/persons_of_concern_demographics_2015-2025.csv')
        
        demo_df = demo_df.rename(columns={
            'Year': 'year',
            'Country of Asylum': 'country_of_asylum',
            'Country of Origin': 'country_of_origin',
            'Total': 'total_people',
            'Female Total': 'female_total',
            'Male Total': 'male_total'
        })
        
        print(f"   ✓ Demographics: {demo_df.shape}")
        
        # 2. Load Conflict Data
        print("   Loading conflict data...")
        conflict_df = pd.read_csv('data/East_Africa_2015_2025_conflictdata.csv')
        conflict_df['date'] = pd.to_datetime(conflict_df['WEEK'], errors='coerce')
        conflict_df['year'] = conflict_df['date'].dt.year
        
        conflict_agg = conflict_df.groupby(['COUNTRY', 'year']).agg({
            'FATALITIES': ['sum', 'mean', 'count'],
            'EVENTS': 'sum'
        }).reset_index()
        
        conflict_agg.columns = ['country', 'year', 'fatalities_sum', 
                               'fatalities_mean', 'fatalities_count', 'events_sum']
        conflict_agg['conflict_intensity'] = conflict_agg['fatalities_sum'] / (conflict_agg['events_sum'] + 1)
        
        print(f"   ✓ Conflict data: {conflict_agg.shape}")
        
        # 3. Load Climate Data
        print("   Loading climate data...")
        climate_df = pd.read_csv('data/Climate_data_sample.csv')
        climate_df = climate_df.rename(columns={
            'County': 'country',
            'Year': 'year'
        })
        print(f"   ✓ Climate data: {climate_df.shape}")
        
        # 4. Load Host Capacity Data
        print("   Loading host capacity data...")
        host_df = pd.read_csv('data/HostCapacity_data_sample.csv')
        host_df = host_df.rename(columns={
            'country': 'country_host',
            'year': 'year_host',
            'num_camps': 'num_camps_host',
            'total_capacity': 'total_capacity_host',
            'estimated_occupancy': 'estimated_occupancy_host'
        })
        host_df['utilization_rate'] = host_df['estimated_occupancy_host'] / (host_df['total_capacity_host'] + 1)
        print(f"   ✓ Host capacity: {host_df.shape}")
        
        # Merge datasets
        print("\n🔗 Merging datasets...")
        df = demo_df.copy()
        
        # Conflict origin
        df = df.merge(conflict_agg, left_on=['country_of_origin', 'year'],
                      right_on=['country', 'year'], how='left')
        df = df.rename(columns={
            'conflict_intensity': 'conflict_intensity_origin',
            'fatalities_sum': 'fatalities_sum_origin',
            'events_sum': 'events_sum_origin'
        })
        
        # Conflict asylum
        df = df.merge(conflict_agg, left_on=['country_of_asylum', 'year'],
                      right_on=['country', 'year'], how='left', suffixes=('', '_asylum'))
        df = df.rename(columns={
            'conflict_intensity': 'conflict_intensity_asylum',
            'fatalities_sum': 'fatalities_sum_asylum',
            'events_sum': 'events_sum_asylum'
        })
        
        # Climate
        df = df.merge(climate_df, left_on=['country_of_origin', 'year'], right_on=['country', 'year'], how='left')
        
        # Host
        df = df.merge(host_df, left_on=['country_of_asylum', 'year'], right_on=['country_host', 'year_host'], how='left')
        
        df = df.drop(['country', 'country_host', 'year_host'], axis=1, errors='ignore')
        print(f"   ✓ Merged dataset: {df.shape}")
        
        # Create features
        print("\n⚙️ Creating features...")
        df = df.sort_values(['country_of_origin', 'country_of_asylum', 'year'])
        df['lag1_total'] = df.groupby(['country_of_origin', 'country_of_asylum'])['total_people'].shift(1)
        df['lag2_total'] = df.groupby(['country_of_origin', 'country_of_asylum'])['total_people'].shift(2)
        df['rolling_mean_3'] = df.groupby(['country_of_origin', 'country_of_asylum'])['total_people'].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        df['rolling_std_3'] = df.groupby(['country_of_origin', 'country_of_asylum'])['total_people'].rolling(3, min_periods=1).std().reset_index(level=[0,1], drop=True)
        df['growth_rate'] = (df['total_people'] - df['lag1_total']) / (df['lag1_total'] + 1)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df.to_csv('data/processed_refugee_data.csv', index=False)
        print(f"   ✓ Processed data saved: {df.shape}")
        
        return df
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None

# -----------------------------
# Train ML models
# -----------------------------
def train_refugee_forecaster(df):
    print("\n🤖 Training ML models on real data...")
    os.makedirs('models', exist_ok=True)
    exclude_cols = ['country_of_origin', 'country_of_asylum', 'year', 'total_people', 'female_total', 'male_total']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
    
    X = df[feature_cols]
    y = df['total_people']
    
    years = sorted(df['year'].unique())
    train_years = years[:-2]
    test_years = years[-2:]
    
    X_train = X[df['year'].isin(train_years)]
    X_test = X[df['year'].isin(test_years)]
    y_train = y[df['year'].isin(train_years)]
    y_test = y[df['year'].isin(test_years)]
    
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                     min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'Random Forest': {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAE_percent': float((mae / y_test.mean()) * 100)
        }
    }
    
    print(f"   Random Forest Performance:")
    print(f"     MAE: {mae:,.0f} refugees ({results['Random Forest']['MAE_percent']:.1f}% of mean)")
    print(f"     RMSE: {rmse:,.0f}")
    print(f"     R²: {r2:.3f}")
    
    joblib.dump(rf_model, 'models/refugee_forecaster_rf.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    model_info = {
        'training_date': datetime.now().isoformat(),
        'training_years': train_years,
        'test_years': test_years,
        'feature_count': len(feature_cols),
        'model_performance': results,
        'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_.tolist()))
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2, default=convert)
    
    return rf_model, feature_cols, results

# -----------------------------
# Generate predictions
# -----------------------------
def generate_predictions(df, model, feature_cols):
    print("\n🔮 Generating forecasts...")
    current_year = df['year'].max()
    next_year = current_year + 1
    predictions = []
    unique_pairs = df[['country_of_origin', 'country_of_asylum']].drop_duplicates()
    
    for origin, asylum in unique_pairs.values:
        pair_data = df[(df['country_of_origin'] == origin) & (df['country_of_asylum'] == asylum)]
        if len(pair_data) > 0:
            latest = pair_data.sort_values('year').iloc[-1]
            features = latest[feature_cols].values.reshape(1, -1)
            try:
                pred = model.predict(features)[0]
                predictions.append({
                    'origin': origin,
                    'asylum': asylum,
                    'current_year': current_year,
                    'current_population': int(latest['total_people']),
                    'predicted_year': next_year,
                    'predicted_population': max(0, int(pred)),
                    'growth_rate': (pred - latest['total_people']) / (latest['total_people'] + 1),
                    'confidence': 'High' if abs(pred - latest['total_people']) < 10000 else 'Medium' if abs(pred - latest['total_people']) < 25000 else 'Low'
                })
            except:
                continue
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv('models/predictions_next_year.csv', index=False)
    
    asylum_predictions = predictions_df.groupby('asylum').agg({
        'predicted_population': 'sum',
        'current_population': 'sum'
    }).reset_index()
    asylum_predictions['growth'] = (asylum_predictions['predicted_population'] - asylum_predictions['current_population']) / asylum_predictions['current_population']
    asylum_predictions.to_csv('models/asylum_predictions.csv', index=False)
    
    return predictions_df, asylum_predictions

# -----------------------------
# Visualization data
# -----------------------------
def create_visualization_data(df, predictions_df):
    print("\n🎨 Creating visualization datasets...")
    time_series = df.groupby(['year', 'country_of_asylum'])['total_people'].sum().reset_index()
    time_series.rename(columns={'country_of_asylum': 'asylum', 'total_people': 'refugees'}, inplace=True)
    time_series.to_csv('models/time_series_data.csv', index=False)
    
    top_asylums = time_series[time_series['year'] == time_series['year'].max()].nlargest(5, 'refugees')['asylum'].tolist()
    
    origin_distribution = []
    for asylum in top_asylums:
        asylum_data = df[df['country_of_asylum'] == asylum]
        latest_year = asylum_data['year'].max()
        latest_data = asylum_data[asylum_data['year'] == latest_year]
        origins = latest_data.groupby('country_of_origin')['total_people'].sum().nlargest(10)
        for origin, count in origins.items():
            origin_distribution.append({
                'asylum': asylum,
                'origin': origin,
                'refugees': int(count),
                'percentage': count / origins.sum() * 100,
                'year': latest_year
            })
    
    pd.DataFrame(origin_distribution).to_csv('models/origin_distribution.csv', index=False)
    
    east_africa_coords = {
        'Kenya': {'lat': 1.0, 'lon': 38.0},
        'Uganda': {'lat': 1.5, 'lon': 32.0},
        'Tanzania': {'lat': -6.0, 'lon': 35.0},
        'Ethiopia': {'lat': 9.0, 'lon': 40.0},
        'Somalia': {'lat': 5.0, 'lon': 48.0},
        'South Sudan': {'lat': 7.0, 'lon': 30.0},
        'Rwanda': {'lat': -2.0, 'lon': 30.0},
        'Burundi': {'lat': -3.5, 'lon': 30.0},
        'DR Congo': {'lat': -4.0, 'lon': 21.0},
        'Djibouti': {'lat': 11.5, 'lon': 43.0},
        'Eritrea': {'lat': 15.0, 'lon': 39.0}
    }
    
    map_data = []
    top_flows = predictions_df.nlargest(15, 'predicted_population')
    for _, flow in top_flows.iterrows():
        if flow['origin'] in east_africa_coords and flow['asylum'] in east_africa_coords:
            map_data.append({
                'origin': flow['origin'],
                'asylum': flow['asylum'],
                'origin_lat': east_africa_coords[flow['origin']]['lat'],
                'origin_lon': east_africa_coords[flow['origin']]['lon'],
                'asylum_lat': east_africa_coords[flow['asylum']]['lat'],
                'asylum_lon': east_africa_coords[flow['asylum']]['lon'],
                'flow_size': min(10, flow['predicted_population'] / 5000),
                'current_flow': flow['current_population'],
                'predicted_flow': flow['predicted_population'],
                'growth': flow['growth_rate']
            })
    
    pd.DataFrame(map_data).to_csv('models/map_flow_data.csv', index=False)
    print("   ✓ Created visualization datasets")
    return True

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    print("=" * 70)
    print("         REFUGEE FLOW FORECASTING - REAL DATA PIPELINE")
    print("=" * 70)
    
    df = load_and_merge_datasets()
    if df is None:
        return
    
    model, feature_cols, results = train_refugee_forecaster(df)
    predictions_df, asylum_predictions = generate_predictions(df, model, feature_cols)
    create_visualization_data(df, predictions_df)
    
    summary = {
        'project': 'Refugee Flow Forecaster',
        'timestamp': datetime.now().isoformat(),
        'data_summary': {
            'total_records': int(len(df)),
            'years_covered': f"{df['year'].min()} to {df['year'].max()}",
            'unique_origins': int(df['country_of_origin'].nunique()),
            'unique_asylums': int(df['country_of_asylum'].nunique()),
            'total_refugees': int(df['total_people'].sum())
        },
        'model_performance': results,
        'prediction_summary': {
            'total_predictions': int(len(predictions_df)),
            'total_predicted_refugees': int(predictions_df['predicted_population'].sum()),
            'top_asylum_countries': asylum_predictions.nlargest(3, 'predicted_population')[['asylum', 'predicted_population']].to_dict('records')
        },
        'files_generated': [
            'models/refugee_forecaster_rf.pkl',
            'models/predictions_next_year.csv',
            'models/asylum_predictions.csv',
            'models/time_series_data.csv',
            'models/origin_distribution.csv',
            'models/map_flow_data.csv',
            'models/model_info.json'
        ]
    }
    
    with open('models/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()

