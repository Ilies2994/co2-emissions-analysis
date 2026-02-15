"""
model.py - Entrainement et comparaison des modèles ML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class CO2PredictionModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
    
    def prepare_data(self, df, target='co2', test_size=0.2):
        """Prépare X, y et split train/test"""
        feature_cols = ['year', 'gdp_per_capita', 'population_millions',
                       'energy_use_per_capita', 'years_since_start']
        
        # ajouter les colonnes optionnelles
        optional = ['carbon_intensity', 'co2_per_capita', 'renewable_energy_pct',
                    'forest_area_pct', 'country_encoded']
        
        # encoder country si pas fait
        if 'country_encoded' not in df.columns and 'country' in df.columns:
            df['country_encoded'] = pd.Categorical(df['country']).codes
        
        for col in optional:
            if col in df.columns:
                feature_cols.append(col)
        
        available = [c for c in feature_cols if c in df.columns]
        
        X = df[available].values
        y = df[target].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # scaling
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, available
    
    def train_all(self, X_train, y_train):
        """Entraine les 3 modèles"""
        print("\nEntrainement des modeles...")
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['Linear Regression'] = lr
        print("  Linear Regression OK")
        
        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=15, 
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        print("  Random Forest OK")
        
        # XGBoost
        xgb = XGBRegressor(
            n_estimators=100, max_depth=7,
            learning_rate=0.1, random_state=42
        )
        xgb.fit(X_train, y_train)
        self.models['XGBoost'] = xgb
        print("  XGBoost OK")
        
        return self.models
    
    def evaluate_all(self, X_test, y_test):
        """Evalue tous les modeles"""
        print("\n--- Resultats ---")
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            self.results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
            print(f"{name:25s} | R2={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")
        
        return self.results
    
    def plot_predictions_comparison(self, y_test):
        """Compare les predictions des modeles"""
        fig, axes = plt.subplots(1, len(self.models), figsize=(5*len(self.models), 5))
        if len(self.models) == 1:
            axes = [axes]
        
        for ax, (name, model) in zip(axes, self.models.items()):
            # on recupere X_test via le scaler (pas ideal mais bon)
            pass
        
        # version simplifiée : juste le bar chart des R2
        names = list(self.results.keys())
        r2s = [self.results[n]['R2'] for n in names]
        
        plt.figure(figsize=(8, 5))
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        plt.bar(names, r2s, color=colors[:len(names)])
        plt.ylabel('R² Score')
        plt.title('Comparaison des modèles')
        plt.ylim(0, 1)
        for i, v in enumerate(r2s):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig('Figures/model_comparison.png', dpi=200)
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """Feature importance du Random Forest"""
        if 'Random Forest' not in self.models:
            return
        
        rf = self.models['Random Forest']
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], color='steelblue')
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('Figures/feature_importance.png', dpi=200)
        plt.show()
