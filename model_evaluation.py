"""
model_evaluation.py - Cross-validation, learning curves et analyse des résidus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve, KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    mean_absolute_error, mean_absolute_percentage_error
)
from scipy import stats


class ModelEvaluator:
    def __init__(self, models_dict, X, y, feature_names):
        self.models = models_dict
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.cv_results = {}
    
    def cross_validation(self, cv=5, scoring='r2'):
        """Cross-validation K-Fold sur tous les modeles"""
        print(f"\nCross-validation ({cv}-Fold)")
        print("-" * 40)
        
        results = []
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=scoring)
            
            result = {
                'Model': name,
                'Mean_Score': scores.mean(),
                'Std_Score': scores.std(),
                'Min_Score': scores.min(),
                'Max_Score': scores.max(),
                'Scores': scores
            }
            results.append(result)
            self.cv_results[name] = result
            
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        df_results = pd.DataFrame(results)[['Model', 'Mean_Score', 'Std_Score', 'Min_Score', 'Max_Score']]
        df_results = df_results.sort_values('Mean_Score', ascending=False)
        
        self._plot_cv_comparison(results)
        return df_results
    
    def _plot_cv_comparison(self, results):
        """Boxplot + barplot des scores CV"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        scores_data = [r['Scores'] for r in results]
        model_names = [r['Model'] for r in results]
        
        bp = axes[0].boxplot(scores_data, labels=model_names, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[0].set_ylabel('R²')
        axes[0].set_title('Distribution des Scores CV')
        axes[0].grid(axis='y', alpha=0.3)
        
        means = [r['Mean_Score'] for r in results]
        stds = [r['Std_Score'] for r in results]
        bars = axes[1].bar(range(len(results)), means, yerr=stds, 
                          capsize=5, color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_xticks(range(len(results)))
        axes[1].set_xticklabels(model_names)
        axes[1].set_ylabel('R² Moyen')
        axes[1].set_title('Performance moyenne')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar, mean in zip(bars, means):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('Figures/cv_comparison.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def learning_curves(self, model, model_name, cv=5, n_points=10):
        """Learning curves pour diagnostiquer biais/variance"""
        print(f"Learning curve: {model_name}...")
        
        train_sizes = np.linspace(0.1, 1.0, n_points)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, self.X, self.y,
            train_sizes=train_sizes, cv=cv, scoring='r2',
            n_jobs=-1, random_state=42
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
                label='Train', linewidth=2)
        plt.fill_between(train_sizes_abs, train_mean - train_std, 
                        train_mean + train_std, alpha=0.15, color='blue')
        
        plt.plot(train_sizes_abs, test_mean, 'o-', color='green', 
                label='Validation', linewidth=2)
        plt.fill_between(train_sizes_abs, test_mean - test_std, 
                        test_mean + test_std, alpha=0.15, color='green')
        
        plt.xlabel("Taille d'entrainement")
        plt.ylabel('Score R²')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'Figures/learning_curve_{model_name.replace(" ", "_")}.png', 
                   dpi=200, bbox_inches='tight')
        plt.show()
        
        gap = train_mean[-1] - test_mean[-1]
        print(f"  Train: {train_mean[-1]:.4f}, Valid: {test_mean[-1]:.4f}, Gap: {gap:.4f}")
        
        return train_sizes_abs, train_mean, test_mean
    
    def residual_analysis(self, model, model_name, X_test, y_test):
        """Analyse des résidus"""
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # residus vs predictions
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predictions')
        axes[0, 0].set_ylabel('Résidus')
        axes[0, 0].set_title('Résidus vs Predictions')
        
        # distribution des residus
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Résidus')
        axes[0, 1].set_title('Distribution des résidus')
        
        # QQ plot
        stats.probplot(residuals, plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # predictions vs réel
        axes[1, 1].scatter(y_test, y_pred, alpha=0.5, s=20)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[1, 1].plot(lims, lims, 'r--')
        axes[1, 1].set_xlabel('Valeurs réelles')
        axes[1, 1].set_ylabel('Predictions')
        axes[1, 1].set_title('Réel vs Prédit')
        
        plt.suptitle(f'Analyse des résidus - {model_name}', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'Figures/residuals_{model_name.replace(" ", "_")}.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        # test de normalité
        _, p_val = stats.shapiro(residuals[:500] if len(residuals) > 500 else residuals)
        print(f"  Shapiro-Wilk p-value: {p_val:.4f}", 
              "(normal)" if p_val > 0.05 else "(non-normal)")
        
        return residuals
    
    def generate_metrics_table(self, X_test, y_test):
        """Tableau de métriques pour tous les modeles"""
        metrics_list = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            metrics_list.append({
                'Model': name,
                'R2': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
            })
        
        df = pd.DataFrame(metrics_list).sort_values('R2', ascending=False)
        print("\nMetriques:")
        print(df.to_string(index=False))
        df.to_csv('Figures/model_metrics.csv', index=False)
        return df


def run_complete_evaluation(models_dict, X_train, X_test, y_train, y_test, feature_names):
    """Evaluation complète des modèles"""
    print("\n" + "=" * 50)
    print("EVALUATION DES MODELES")
    print("=" * 50)
    
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    
    evaluator = ModelEvaluator(models_dict, X_full, y_full, feature_names)
    
    cv_results = evaluator.cross_validation(cv=5)
    
    for name, model in models_dict.items():
        evaluator.learning_curves(model, name, cv=5)
    
    # residus sur le meilleur modele
    best_name = cv_results.iloc[0]['Model']
    best_model = models_dict[best_name]
    evaluator.residual_analysis(best_model, best_name, X_test, y_test)
    
    metrics_df = evaluator.generate_metrics_table(X_test, y_test)
    
    print(f"\nMeilleur modele: {best_name} (R2 CV: {cv_results.iloc[0]['Mean_Score']:.4f})")
    return evaluator, cv_results, metrics_df
