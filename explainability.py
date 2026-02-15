"""
explainability.py - Analyse SHAP pour l'explicabilité des modèles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor


class ModelExplainer:
    def __init__(self, model, X_train, X_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
    
    def compute_shap_values(self, use_tree_explainer=True):
        """Calcule les valeurs SHAP"""
        print("Calcul des valeurs SHAP...")
        
        if use_tree_explainer:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict, 
                shap.sample(self.X_train, 100)
            )
        
        self.shap_values = self.explainer.shap_values(self.X_test)
        print(f"SHAP calculé pour {len(self.X_test)} observations")
        return self.shap_values
    
    def plot_summary(self, max_display=10):
        """Summary plot SHAP"""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, self.X_test,
            feature_names=self.feature_names,
            max_display=max_display, show=False
        )
        plt.title('SHAP Summary Plot - Impact des features')
        plt.tight_layout()
        plt.savefig('Figures/shap_summary.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self):
        """Bar plot importance SHAP"""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, self.X_test,
            feature_names=self.feature_names,
            plot_type="bar", show=False
        )
        plt.title('Importance globale des features (SHAP)')
        plt.tight_layout()
        plt.savefig('Figures/shap_importance.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def plot_dependence(self, feature_idx=0):
        """Dependence plot pour une feature"""
        feature_name = self.feature_names[feature_idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, self.shap_values, self.X_test,
            feature_names=self.feature_names, show=False
        )
        plt.title(f'Dependence Plot : {feature_name}')
        plt.tight_layout()
        plt.savefig(f'Figures/shap_dep_{feature_name}.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def explain_single_prediction(self, idx=0):
        """Explique une prediction individuelle"""
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        prediction = self.model.predict(self.X_test[idx:idx+1])[0]
        print(f"\nPrediction #{idx}: {prediction:.2f} (base: {base_value:.2f})")
        
        # top contributions
        contributions = list(zip(self.feature_names, self.shap_values[idx]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feat, val in contributions[:5]:
            sign = "+" if val > 0 else ""
            print(f"  {feat:25s} : {sign}{val:.2f}")
        
        # waterfall
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[idx],
                base_values=base_value,
                data=self.X_test[idx],
                feature_names=self.feature_names
            ), show=False
        )
        plt.tight_layout()
        plt.savefig(f'Figures/shap_waterfall_{idx}.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        return contributions
    
    def get_feature_importance_df(self):
        importance = np.abs(self.shap_values).mean(axis=0)
        return pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': importance
        }).sort_values('SHAP_Importance', ascending=False)
    
    def compare_with_model_importance(self, model_importance):
        """Compare SHAP vs importance native du modele"""
        shap_imp = self.get_feature_importance_df()
        
        comparison = shap_imp.copy()
        comparison['Model_Importance'] = [
            model_importance[self.feature_names.index(f)] 
            for f in comparison['Feature']
        ]
        
        print("\nComparaison SHAP vs Model:")
        print(comparison.to_string(index=False))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].barh(range(len(shap_imp)), shap_imp['SHAP_Importance'], color='steelblue')
        axes[0].set_yticks(range(len(shap_imp)))
        axes[0].set_yticklabels(shap_imp['Feature'])
        axes[0].set_title('Importance SHAP')
        axes[0].invert_yaxis()
        
        model_sorted = comparison.sort_values('Model_Importance', ascending=True)
        axes[1].barh(range(len(model_sorted)), model_sorted['Model_Importance'], color='coral')
        axes[1].set_yticks(range(len(model_sorted)))
        axes[1].set_yticklabels(model_sorted['Feature'])
        axes[1].set_title('Importance Native')
        
        plt.suptitle('SHAP vs Importance du modèle')
        plt.tight_layout()
        plt.savefig('Figures/shap_vs_model.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        return comparison


def run_explainability_analysis(model, X_train, X_test, feature_names):
    """Lance l'analyse SHAP complète"""
    print("\n" + "=" * 50)
    print("ANALYSE SHAP")
    print("=" * 50)
    
    explainer = ModelExplainer(model, X_train, X_test, feature_names)
    explainer.compute_shap_values()
    
    explainer.plot_summary()
    explainer.plot_feature_importance()
    
    # dependence plots pour les 3 features les plus importantes
    importance_df = explainer.get_feature_importance_df()
    top_feats = importance_df.head(3)['Feature'].tolist()
    for feat in top_feats:
        idx = feature_names.index(feat)
        explainer.plot_dependence(feature_idx=idx)
    
    # quelques predictions individuelles
    for i in [0, len(X_test)//2, len(X_test)-1]:
        explainer.explain_single_prediction(idx=i)
    
    if hasattr(model, 'feature_importances_'):
        explainer.compare_with_model_importance(model.feature_importances_)
    
    print("Analyse SHAP terminée")
    return explainer
