"""
app.py - Dashboard Streamlit pour l'analyse des émissions CO2
Lancer avec: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CO2 Emissions Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(filepath='data/co2_emissions.csv'):
    """Charge les données"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.warning("Fichier non trouvé, utilisation de données synthétiques")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Données de démo si le CSV n'est pas trouvé"""
    np.random.seed(42)
    years = range(1990, 2024)
    countries = ['USA', 'China', 'India', 'Germany', 'France', 'UK', 'Brazil', 
                'Canada', 'Japan', 'Russia', 'South Africa', 'Australia',
                'Italy', 'Spain', 'Mexico', 'Indonesia', 'South Korea']
    
    data = []
    for country in countries:
        base = np.random.uniform(200, 10000)
        trend = np.random.uniform(-50, 100)
        for year in years:
            gdp = np.random.uniform(500, 20000) * (1 + 0.02 * (year - 1990))
            pop = np.random.uniform(10, 1400) * (1 + 0.01 * (year - 1990))
            energy = np.random.uniform(1000, 15000) * (1 + 0.015 * (year - 1990))
            co2 = max(0, base + trend * (year-1990) + 0.3*gdp + 0.5*energy + np.random.normal(0, 500))
            data.append({
                'country': country, 'year': year, 'co2': co2,
                'gdp': gdp * pop, 'population': pop * 1e6,
                'primary_energy_consumption': energy,
            })
    return pd.DataFrame(data)


@st.cache_data
def prepare_features(df):
    """Feature engineering pour le dashboard"""
    df = df.copy()
    df['population_millions'] = df['population'] / 1e6
    df['gdp_per_capita'] = df['gdp'] / df['population']
    df['co2_per_capita'] = df['co2'] / df['population_millions']
    df['energy_per_capita'] = df['primary_energy_consumption'] / df['population_millions']
    df['years_since_start'] = df['year'] - df['year'].min()
    df['country_encoded'] = pd.Categorical(df['country']).codes
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


@st.cache_resource
def train_models(X_train, y_train):
    models = {}
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    xgb = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    
    return models


# --- Sidebar ---
st.sidebar.markdown("# 🌍 CO2 Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Exploration", "Visualisations", "Modélisation", "Prédictions", "A propos"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Charger vos données")
uploaded_file = st.sidebar.file_uploader("Fichier CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Fichier chargé")
else:
    df = load_data()

df_prepared = prepare_features(df)


# --- Pages ---

if page == "Accueil":
    st.markdown('<h1 class="main-header">🌍 Analyse des Émissions de CO2</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Dashboard interactif pour l'analyse et la prédiction des émissions de CO2 mondiales 
    à l'aide de techniques de Machine Learning.
    
    **Fonctionnalités :**
    - Exploration et visualisation des données
    - Modélisation ML (Linear Regression, Random Forest, XGBoost)
    - Prédictions interactives
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pays", df['country'].nunique())
    col2.metric("Années", f"{df['year'].min()} - {df['year'].max()}")
    col3.metric("Observations", f"{len(df):,}")
    col4.metric("Variables", df.shape[1])


elif page == "Exploration":
    st.header("Exploration des données")
    
    tab1, tab2, tab3 = st.tabs(["Aperçu", "Statistiques", "Valeurs manquantes"])
    
    with tab1:
        st.dataframe(df.head(20))
    with tab2:
        st.dataframe(df.describe())
    with tab3:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.bar_chart(missing)
        else:
            st.success("Pas de valeurs manquantes")


elif page == "Visualisations":
    st.header("Visualisations")
    
    viz_type = st.selectbox("Type de graphique", 
                           ["Evolution temporelle", "Top émetteurs", "Corrélation", "CO2 vs PIB"])
    
    if viz_type == "Evolution temporelle":
        top_n = st.slider("Nombre de pays", 3, 15, 8)
        top_countries = df_prepared.groupby('country')['co2'].mean().nlargest(top_n).index
        data_plot = df_prepared[df_prepared['country'].isin(top_countries)]
        
        fig = px.line(data_plot, x='year', y='co2', color='country',
                     title=f'Evolution CO2 - Top {top_n} pays')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Top émetteurs":
        year_sel = st.selectbox("Année", sorted(df['year'].unique(), reverse=True))
        data_year = df_prepared[df_prepared['year'] == year_sel].nlargest(15, 'co2')
        
        fig = px.bar(data_year, x='co2', y='country', orientation='h',
                    title=f'Top 15 émetteurs ({year_sel})', color='co2',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Corrélation":
        numeric = df_prepared.select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
    
    elif viz_type == "CO2 vs PIB":
        fig = px.scatter(df_prepared, x='gdp_per_capita', y='co2', 
                        color='country', hover_name='country',
                        title='CO2 vs PIB par habitant', opacity=0.5)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


elif page == "Modélisation":
    st.header("Modélisation ML")
    
    features = ['year', 'gdp_per_capita', 'population_millions',
               'energy_per_capita', 'years_since_start', 'country_encoded']
    available = [f for f in features if f in df_prepared.columns]
    
    X = df_prepared[available].values
    y = df_prepared['co2'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = train_models(X_train_s, y_train)
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_s)
        results[name] = {
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
    
    st.subheader("Résultats")
    df_results = pd.DataFrame(results).T
    st.dataframe(df_results.style.highlight_max(axis=0, subset=['R2']))
    
    # bar chart R2
    fig = px.bar(x=list(results.keys()), 
                y=[r['R2'] for r in results.values()],
                title='Comparaison R²', labels={'x': 'Modèle', 'y': 'R²'})
    st.plotly_chart(fig, use_container_width=True)


elif page == "Prédictions":
    st.header("Prédictions")
    
    col1, col2 = st.columns(2)
    with col1:
        year_input = st.number_input("Année", 1990, 2050, 2025)
        gdp_input = st.number_input("PIB/hab ($)", 100, 100000, 10000)
        pop_input = st.number_input("Population (millions)", 0.1, 1500.0, 50.0)
    with col2:
        energy_input = st.number_input("Energie/hab", 10, 50000, 5000)
    
    if st.button("Prédire"):
        features_input = [year_input, gdp_input, pop_input, energy_input, 
                         year_input - 1990, 0]
        
        features = ['year', 'gdp_per_capita', 'population_millions',
                    'energy_per_capita', 'years_since_start', 'country_encoded']
        available = [f for f in features if f in df_prepared.columns]
        
        X_all = df_prepared[available].values
        y_all = df_prepared['co2'].values
        
        scaler = StandardScaler()
        scaler.fit(X_all)
        
        input_scaled = scaler.transform([features_input[:len(available)]])
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        rf.fit(scaler.transform(X_all), y_all)
        
        pred = rf.predict(input_scaled)[0]
        st.success(f"Émissions prédites : **{pred:,.2f} Mt CO2**")


elif page == "A propos":
    st.header("A propos du projet")
    st.markdown("""
    **Projet** : Analyse et prédiction des émissions de CO2 mondiales
    
    **Technologies** : Python, Scikit-learn, XGBoost, SHAP, Streamlit, Plotly
    
    **Dataset** : Our World in Data - CO2 Emissions
    
    **Auteur** : CHALABI Mohammed Ilies  
    4ème année Ingénieur d'Etat en Informatique - spécialité IA  
    Université Djilali Liabes, Sidi Bel Abbes
    """)
