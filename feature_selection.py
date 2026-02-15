"""
feature_selection.py - Méthodes de sélection de features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.results = {}
    
    def correlation_analysis(self, threshold=0.8):
        """Analyse de corrélation + détection de multicolinéarité"""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        corr = df.corr().abs()
        
        # features très corrélées entre elles
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [(col, row, corr.loc[row, col]) 
                     for col in upper.columns 
                     for row in upper.index 
                     if upper.loc[row, col] > threshold]
        
        if high_corr:
            print(f"Features corrélées (>{threshold}):")
            for f1, f2, val in high_corr:
                print(f"  {f1} - {f2}: {val:.3f}")
        
        self.results['correlation'] = high_corr
        return high_corr
    
    def univariate_selection(self, k=5):
        """SelectKBest avec f_regression"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # F-test
        selector_f = SelectKBest(f_regression, k=min(k, len(self.feature_names)))
        selector_f.fit(X_scaled, self.y)
        scores_f = selector_f.scores_
        
        # Mutual Information
        mi_scores = mutual_info_regression(X_scaled, self.y, random_state=42)
        
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'F_Score': scores_f,
            'MI_Score': mi_scores
        }).sort_values('F_Score', ascending=False)
        
        print("\nUnivariate Feature Selection:")
        print(results.to_string(index=False))
        
        self.results['univariate'] = results
        return results
    
    def rfe_selection(self, n_features=5):
        """Recursive Feature Elimination avec Random Forest"""
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        rfe = RFE(rf, n_features_to_select=min(n_features, len(self.feature_names)))
        rfe.fit(X_scaled, self.y)
        
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'Selected': rfe.support_,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        print(f"\nRFE - Top {n_features} features:")
        selected = results[results['Selected']]['Feature'].tolist()
        print(f"  {selected}")
        
        self.results['rfe'] = results
        return results
    
    def tree_importance(self):
        """Importance basée sur Random Forest"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        rf.fit(X_scaled, self.y)
        
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nRandom Forest Feature Importance:")
        print(results.to_string(index=False))
        
        self.results['tree'] = results
        return results
    
    def plot_comparison(self):
        """Compare toutes les méthodes"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        if 'univariate' in self.results:
            data = self.results['univariate'].sort_values('F_Score', ascending=True)
            axes[0].barh(data['Feature'], data['F_Score'], color='steelblue')
            axes[0].set_title('F-Score (Univariate)')
        
        if 'univariate' in self.results:
            data = self.results['univariate'].sort_values('MI_Score', ascending=True)
            axes[1].barh(data['Feature'], data['MI_Score'], color='coral')
            axes[1].set_title('Mutual Information')
        
        if 'tree' in self.results:
            data = self.results['tree'].sort_values('Importance', ascending=True)
            axes[2].barh(data['Feature'], data['Importance'], color='forestgreen')
            axes[2].set_title('RF Importance')
        
        plt.suptitle('Comparaison des méthodes de sélection', fontsize=13)
        plt.tight_layout()
        plt.savefig('Figures/feature_selection_comparison.png', dpi=200, bbox_inches='tight')
        plt.show()


def run_complete_feature_analysis(X, y):
    """Lance toute l'analyse de sélection de features"""
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_arr = X
    
    print("=" * 50)
    print("ANALYSE DE SELECTION DE FEATURES")
    print("=" * 50)
    
    selector = FeatureSelector(X_arr, y, feature_names)
    
    selector.correlation_analysis()
    selector.univariate_selection(k=5)
    selector.rfe_selection(n_features=5)
    selector.tree_importance()
    selector.plot_comparison()
    
    print("\nAnalyse terminée")
    return selector
