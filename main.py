"""
main.py - Script principal du pipeline d'analyse CO2
Auteur: CHALABI Mohammed Ilies
"""
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np

from data_processing import CO2DataProcessor
from visualizations import CO2Visualizer
from model import CO2PredictionModel

# modules optionnels
try:
    from feature_selection import run_complete_feature_analysis
    HAS_FEATURE_SEL = True
except ImportError:
    HAS_FEATURE_SEL = False

try:
    from dimensionality_reduction import analyze_pca
    HAS_PCA = True
except ImportError:
    HAS_PCA = False

try:
    from model_evaluation import run_complete_evaluation
    HAS_EVAL = True
except ImportError:
    HAS_EVAL = False

try:
    from explainability import run_explainability_analysis
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def main():
    print("=" * 60)
    print("  ANALYSE ET PREDICTION DES EMISSIONS DE CO2")
    print("=" * 60)
    
    os.makedirs('Figures', exist_ok=True)
    
    # 1. Chargement et nettoyage
    data_path = 'data/co2_emissions.csv'
    
    processor = CO2DataProcessor(data_path)
    df = processor.load_data()
    processor.explore_data()
    df_clean = processor.clean_data()
    df_final = processor.feature_engineering()
    
    print(f"\nDataset pret: {len(df_final)} lignes, {len(df_final.columns)} colonnes")
    
    # 2. Feature Selection
    if HAS_FEATURE_SEL:
        print("\n" + "=" * 60)
        print("  SELECTION DE FEATURES")
        print("=" * 60)
        
        feature_cols = ['year', 'gdp_per_capita', 'population_millions', 
                       'energy_use_per_capita', 'years_since_start']
        
        if 'renewable_energy_pct' in df_final.columns:
            feature_cols.append('renewable_energy_pct')
        if 'forest_area_pct' in df_final.columns:
            feature_cols.append('forest_area_pct')
        
        df_final['country_encoded'] = pd.Categorical(df_final['country']).codes
        feature_cols.append('country_encoded')
        
        available = [c for c in feature_cols if c in df_final.columns]
        
        if len(available) >= 4:
            X_analysis = df_final[available]
            y_analysis = df_final['co2']
            try:
                selector = run_complete_feature_analysis(X_analysis, y_analysis)
            except Exception as e:
                print(f"Erreur feature selection: {e}")
    
    # 3. PCA
    if HAS_PCA:
        print("\n" + "=" * 60)
        print("  REDUCTION DE DIMENSIONNALITE (PCA)")
        print("=" * 60)
        
        pca_features = ['gdp_per_capita', 'population_millions', 
                       'energy_use_per_capita', 'carbon_intensity', 'co2_per_capita']
        
        for feat in ['renewable_energy_pct', 'forest_area_pct']:
            if feat in df_final.columns:
                pca_features.append(feat)
        
        available_pca = [c for c in pca_features if c in df_final.columns]
        
        if len(available_pca) >= 3:
            try:
                X_reduced, pca_model, components = analyze_pca(
                    df_final[available_pca], feature_names=available_pca
                )
            except Exception as e:
                print(f"Erreur PCA: {e}")
    
    # 4. Visualisations
    print("\n" + "=" * 60)
    print("  VISUALISATIONS")
    print("=" * 60)
    
    viz = CO2Visualizer(df_final)
    viz.plot_all()
    
    # 5. Modélisation
    print("\n" + "=" * 60)
    print("  MODELISATION ML")
    print("=" * 60)
    
    model = CO2PredictionModel()
    X_train, X_test, y_train, y_test, features = model.prepare_data(df_final)
    model.train_all(X_train, y_train)
    model.evaluate_all(X_test, y_test)
    model.plot_predictions_comparison(y_test)
    model.plot_feature_importance(features)
    
    # 6. Evaluation
    if HAS_EVAL:
        print("\n" + "=" * 60)
        print("  EVALUATION (Cross-Validation)")
        print("=" * 60)
        
        try:
            evaluator, cv_results, metrics_df = run_complete_evaluation(
                models_dict=model.models,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                feature_names=features
            )
        except Exception as e:
            print(f"Erreur evaluation: {e}")
    
    # 7. SHAP
    if HAS_SHAP:
        print("\n" + "=" * 60)
        print("  EXPLICABILITE (SHAP)")
        print("=" * 60)
        
        try:
            rf_model = model.models['Random Forest']
            explainer = run_explainability_analysis(
                model=rf_model,
                X_train=X_train, X_test=X_test,
                feature_names=features
            )
        except Exception as e:
            print(f"Erreur SHAP: {e}")
    
    print("\n" + "=" * 60)
    print("  ANALYSE TERMINEE")
    print("=" * 60)
    
    return df_final, model


if __name__ == "__main__":
    df, model = main()
