"""
dimensionality_reduction.py - PCA sur les données CO2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def analyze_pca(X, feature_names=None, n_components=None):
    """
    Applique la PCA et génère les visualisations
    Retourne X_reduced, le modèle PCA, et les composantes
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # standardiser avant PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA complète d'abord pour voir la variance expliquée
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # choisir n_components pour >= 95% de variance
    if n_components is None:
        n_components = np.argmax(cumulative_var >= 0.95) + 1
        n_components = max(2, n_components)  # au moins 2 pour la visu
    
    print(f"PCA: {X.shape[1]} features -> {n_components} composantes")
    print(f"Variance expliquée: {cumulative_var[n_components-1]*100:.1f}%")
    
    # PCA finale
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    # -- Graphiques --
    
    # 1. Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
                pca_full.explained_variance_ratio_, alpha=0.7, label='Individuelle')
    axes[0].plot(range(1, len(cumulative_var)+1), cumulative_var, 
                'ro-', label='Cumulée')
    axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Composante')
    axes[0].set_ylabel('Variance expliquée')
    axes[0].set_title('Scree Plot')
    axes[0].legend()
    
    # 2. Projection sur PC1-PC2
    axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.3, s=10)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('Projection PCA')
    
    plt.tight_layout()
    plt.savefig('Figures/pca_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # 3. Biplot - contribution des features
    plt.figure(figsize=(8, 8))
    components = pca.components_[:2]
    
    for i, feat in enumerate(feature_names):
        plt.arrow(0, 0, components[0, i], components[1, i],
                 head_width=0.02, head_length=0.01, fc='red', ec='red', alpha=0.7)
        plt.text(components[0, i]*1.15, components[1, i]*1.15, feat, fontsize=9)
    
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
    plt.gca().add_patch(circle)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Cercle des corrélations')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figures/pca_biplot.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # afficher les loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    print(f"\nLoadings (contributions aux composantes):")
    print(loadings.round(3))
    
    return X_reduced, pca, pca.components_
