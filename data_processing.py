"""
data_processing.py - Chargement et nettoyage des données CO2
"""

import pandas as pd
import numpy as np


class CO2DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_clean = None
    
    def load_data(self):
        """Charge le CSV"""
        self.df = pd.read_csv(self.filepath)
        print(f"Données chargées : {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Affiche les infos de base sur le dataset"""
        print("\n--- Exploration ---")
        print(f"Shape: {self.df.shape}")
        print(f"\nTypes:\n{self.df.dtypes}")
        print(f"\nValeurs manquantes:\n{self.df.isnull().sum()}")
        print(f"\nStats:\n{self.df.describe()}")
    
    def clean_data(self):
        """Nettoie les données : suppression NaN, filtrage"""
        df = self.df.copy()
        
        # colonnes essentielles
        cols_needed = ['country', 'year', 'co2', 'population', 'gdp', 
                       'primary_energy_consumption']
        available = [c for c in cols_needed if c in df.columns]
        df = df[available]
        
        # supprimer les lignes avec trop de NaN
        df = df.dropna(subset=['co2', 'country', 'year'])
        
        # garder que les pays (pas les continents/world)
        exclude = ['World', 'Asia', 'Europe', 'Africa', 'North America', 
                   'South America', 'Oceania', 'European Union (27)',
                   'High-income countries', 'Low-income countries',
                   'Upper-middle-income countries', 'Lower-middle-income countries']
        df = df[~df['country'].isin(exclude)]
        
        # remplir les NaN restantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # enlever co2 <= 0
        df = df[df['co2'] > 0]
        
        self.df_clean = df
        print(f"Après nettoyage : {df.shape}")
        return df
    
    def feature_engineering(self):
        """Crée de nouvelles features"""
        df = self.df_clean.copy()
        
        # features par habitant
        df['population_millions'] = df['population'] / 1e6
        df['gdp_per_capita'] = df['gdp'] / df['population']
        df['co2_per_capita'] = df['co2'] / df['population_millions']
        
        if 'primary_energy_consumption' in df.columns:
            df['energy_use_per_capita'] = df['primary_energy_consumption'] / df['population_millions']
            # intensité carbone = co2 / énergie
            df['carbon_intensity'] = df['co2'] / df['primary_energy_consumption'].replace(0, np.nan)
        
        # variable temporelle
        df['years_since_start'] = df['year'] - df['year'].min()
        
        # renewable et forest si dispo
        for col in ['renewable_energy_pct', 'forest_area_pct']:
            if col not in df.columns:
                pass  # on skip, pas grave
        
        # nettoyer les inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        self.df_clean = df
        print(f"Après feature engineering : {df.shape}")
        return df
