"""
visualizations.py - Graphiques pour l'analyse CO2
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class CO2Visualizer:
    def __init__(self, df):
        self.df = df
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_emissions_over_time(self, top_n=10):
        """Evolution des émissions pour les top N pays"""
        top_countries = (self.df.groupby('country')['co2']
                        .mean().nlargest(top_n).index)
        
        plt.figure(figsize=(12, 6))
        for country in top_countries:
            data = self.df[self.df['country'] == country]
            plt.plot(data['year'], data['co2'], label=country, linewidth=1.5)
        
        plt.xlabel('Année')
        plt.ylabel('Émissions CO2 (Mt)')
        plt.title(f'Evolution des émissions - Top {top_n} pays')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig('Figures/emissions_over_time.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def plot_top_emitters(self, year=2020):
        """Bar chart des plus gros émetteurs pour une année donnée"""
        # prendre l'année la plus proche si pas dispo
        available_years = self.df['year'].unique()
        if year not in available_years:
            year = max(available_years)
        
        data_year = self.df[self.df['year'] == year]
        top = data_year.nlargest(15, 'co2')
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top)))
        plt.barh(top['country'], top['co2'], color=colors)
        plt.xlabel('Émissions CO2 (Mt)')
        plt.title(f'Top 15 émetteurs de CO2 ({year})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('Figures/top_emitters.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self):
        """Matrice de corrélation"""
        numeric = self.df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True)
        plt.title('Matrice de corrélation')
        plt.tight_layout()
        plt.savefig('Figures/correlation_matrix.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def plot_co2_vs_gdp(self):
        """Scatter CO2 vs PIB par habitant"""
        plt.figure(figsize=(10, 6))
        
        if 'gdp_per_capita' in self.df.columns:
            plt.scatter(self.df['gdp_per_capita'], self.df['co2'], 
                       alpha=0.3, s=10)
            plt.xlabel('PIB par habitant ($)')
            plt.ylabel('CO2 (Mt)')
            plt.title('Relation CO2 - PIB par habitant')
            plt.tight_layout()
            plt.savefig('Figures/co2_vs_gdp.png', dpi=200, bbox_inches='tight')
            plt.show()
        else:
            print("gdp_per_capita pas disponible")
    
    def plot_all(self):
        """Lance toutes les visualisations"""
        print("Generation des graphiques...")
        self.plot_emissions_over_time()
        self.plot_top_emitters()
        self.plot_correlation_matrix()
        self.plot_co2_vs_gdp()
        print("Graphiques terminés")
