import streamlit as st
import numpy as np
import pandas as pd 
import time
import matplotlib.pyplot as plt
from numpy import nan
import seaborn as sns

sns.set_style("darkgrid")



def log(func):
    def wrapper(*args,**kwargs):
        with open("log.txt","a") as f:
            debut = time.time()
            value = func(*args,**kwargs)
            fin = time.time()
            f.write("\nCalled function "+ func.__name__ + " in "+ str(fin - debut)+"\n")
            return value
    return wrapper



st.title("Analyser et comprendre les enjeux de l'investissement immobilier")
st.subheader("Cette application propose des visualisations de données variées pour investir plus sereinement dans l'immobilier. Récemment rendu public, les data set utilisés sont accessibles via le site data.gouv.fr : https://www.data.gouv.fr/en/datasets/demandes-de-valeurs-foncieres/")
# 1. Data Loading

@log
def load_dataset(path):
    df = pd.read_csv(path)
    return df


def my_sidebar():
    st.sidebar.title("Options")

@log
def get_int(dt):
    if dt =="NaN":
        dt.dropna(subset = ["longitude"], inplace=True)
        dt.dropna(subset = ["latitude"], inplace=True)
    else:
        return dt

# 2. Explore and Process : Data cleaning, pre-processing, transformation and enrichment
def transform_data(df):
    df['adresse_code_voie'] = df['adresse_code_voie'].astype(str)
    df['code_commune'] = df['code_commune'].astype(str)
    df['code_departement'] = df['code_departement'].astype(str)
    df['numero_volume'] = df['numero_volume'].astype(str)
    return df


@log
def pre_processing(df):
    df.drop(subset = ["lot1_numero", "lot2_numero", "lot3_numero", "lot4_numero", "lot5_numero", "lot1_surface_carrez", "lot2_surface_carrez", "lot3_surface_carrez", "lot4_surface_carrez", "lot5_surface_carrez", "adresse_suffixe", "ancien_nom_commune", "ancien_code_commune", "ancien_id_parcelle", "code_nature_culture", "nature_culture_speciale", "code_nature_culture_speciale", "numero_volume"],  inplace=True)
    return df


@log
def price_per_meter(df):
    df['prix_par_metre_carre'] = df['valeur_fonciere'] / df['surface_terrain'] 
    return ['prix_par_metre_carre']

# 3. Data visualization : visual reprenstation and analysis through different axes and aggregations

@log
#Scatter plot _ Departement / prix par metre carré
def scatter_plot(df):
    if st.sidebar.checkbox("Scatter plot : Prix au metre carré en fonction du département"):
        gb = df.groupby(['code_departement'])
        by_departement = gb.apply(count_rows).nlargest(20).rename("nb_transaction")
        by_dep_price = gb.prix_par_metre_carre.mean().rename("moy_prix_metre_carre")
        df3 = pd.merge(by_departement, by_dep_price, left_index=True, right_index=True)
        st.write(df3)
        st.subheader('Prix par metre carré dans les 20 departements où il y a le plus de tansactions _ 2020')
        figure = sns.relplot(x= df3.nb_transaction, y = df3.moy_prix_metre_carre) 
        st.pyplot(figure)

@log
#Pie chart _ Repartition Type local
def pie_chart(df):
    if st.sidebar.checkbox("Pie chart : Repartition Type local"):
        gb = df.groupby(['type_local'])
        by_type_local = gb.apply(count_rows).rename("nb_by_type_local")
        st.write(by_type_local)
        myexplode = [0, 0, 0, 0.2]
        plt.subplot()
        plt.pie(by_type_local, labels = ("Appartement", "Dependance", "Local industriel. commercial ou assimilé", "Maison"), explode = myexplode, shadow = True)
        plt.title('Répartition des types de local en 2020')
        st.pyplot(fig=plt)

@log
#internal bar chart _ Prix moyen en fonction du type de local
def bar_chart(df,df2):
    st.subheader("Prix moyen en fonction du type de local")
    gb = df.groupby(['type_local'])
    gb_2019 = df2.groupby(['type_local'])
    by_type_local = gb.apply(count_rows).rename("nb_by_type_local")
    by_type_price_2020 = gb.prix_par_metre_carre.mean().rename("moy_prix_metre_carre_2020")
    by_type_price_2019 = gb_2019.prix_par_metre_carre.mean().rename("moy_prix_metre_carre_2019")
    df4 = pd.merge(by_type_price_2020,by_type_price_2019, left_index=True, right_index=True)
    st.write(df4)
    chart_data = df4[["moy_prix_metre_carre_2020", "moy_prix_metre_carre_2019"]]
    st.bar_chart(chart_data)

@log
def display_table(df, text):
    if st.checkbox(text):
        st.subheader('Raw data')
        st.write(df)

@log
def moy_rows(rows):
    return avg(rows)

@log
def count_rows(rows): 
    return len(rows)

@log
def prix_square_metter_rows(rows): 
    return np.mean(rows)

@log
#Histogramme : Transaction immobilière par department
def display_histo_one(df):
    if st.sidebar.checkbox("Histogramme : Transaction immobilière par department"):
        by_departement = df.groupby('code_departement').apply(count_rows)
        st.subheader('Nombre de transactions immobilière par depatement (Top 15) -- France _ 2020')
        plt.bar(range(1, 16), by_departement.nlargest(15))
        plt.xticks(range(1, 16), by_departement.nlargest(15).index)
        plt.xlabel('Departement')
        plt.ylabel('Nombre de transactions immobilière')
        st.pyplot(fig=plt)

@log
# Map : Répartition géographique des biens 
def display_map(df):
    #delete the NaN values for longitude and latitude
    df.dropna(subset = ["latitude"], inplace=True)
    df.dropna(subset = ["longitude"], inplace=True)
    surface_to_filter = st.sidebar.slider('Surface du bien : ', 9, 300, 80)
    type_to_filter = st.sidebar.selectbox("Choisissez un type de bien :", ["Maison", "Dépendance", "Appartement", "Local industriel. commercial ou assimilé" ])
    filtered_data = df[(df["surface_reelle_bati"] == surface_to_filter) & (df["type_local"] == type_to_filter)]
    st.subheader('Situation géographique des %ss de %s mètre carré' % (type_to_filter, surface_to_filter))
    st.map(filtered_data)

# 4. Insights extraction to support analytical findings and decision making


def main():
    my_sidebar()
    path_data_2020="data/full_2020.csv"
    path_data_2019="data/full_2019.csv"
    df_2020 = load_dataset(path_data_2020)
    df_2019 = load_dataset(path_data_2019)
    #sampling data 
    df_2020_sample = df_2020.sample(n=100000, random_state=1)
    df_2019_sample = df_2019.sample(n=100000, random_state=1)
    price_per_meter(df_2020)
    price_per_meter(df_2019)
    #2.
    #pre_processing(df_2020)
    transform_data(df_2020)
    #3.
    display_table(df_2020_sample, "Voir un echantillons du data set 2020")
    display_map(df_2020)
    st.sidebar.text("Autres visualisations")
    display_histo_one(df_2020)
    scatter_plot(df_2020)
    pie_chart(df_2020)
    bar_chart(df_2020,df_2019)


main()