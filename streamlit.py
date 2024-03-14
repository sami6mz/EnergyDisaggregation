import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import datetime
import shap
import streamlit as st
from carbon_models import *
from sklearn.metrics import r2_score, mean_absolute_percentage_error

############ Partie désagrégation ############

st.title("Capstone Project : Energy Disaggregation for CO2 forecasting")

region = st.selectbox(
    'Choisir la région',
    ('Toute la France', 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
        'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
        'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
        "Provence-Alpes-Côte d'Azur", 'Île-de-France')
)

col11, col12 = st.columns([1, 1])

st.header("Désaggrégation de la courbe de charge")

# Years selection boxes
with col11:
    begin_year = st.number_input("Année de départ", value=2019, placeholder="2019", min_value=2019, max_value=2023)
with col12:
    end_year = st.number_input("Année de fin", value=2022, placeholder="2023", min_value=begin_year, max_value=2023)

# Afficher la courbe de consommation d'énergie désagrégée de begin_year à end_year
courbe_desag = "../Data/ctr_regions_2022.csv"
conso_reelle = "../Data/df_process_2022.csv"

courbe_d = pd.read_csv(courbe_desag, sep=",")
conso_r = pd.read_csv(conso_reelle, sep=",")
courbe_d = pd.DataFrame(courbe_d)
conso_r = pd.DataFrame(conso_r)

# Garder les années de begin_year à end_year
courbe_d['Date - Heure'] = pd.to_datetime(courbe_d['Date - Heure'])
conso_r['Date - Heure'] = pd.to_datetime(conso_r['Date - Heure'])
courbe_d_filter = courbe_d[(courbe_d['Date - Heure'].dt.year >= begin_year) & (courbe_d['Date - Heure'].dt.year <= end_year)]
conso_r_filter = conso_r[(conso_r['Date - Heure'].dt.year >= begin_year) & (conso_r['Date - Heure'].dt.year <= end_year)]

if region!='Toute la France' :
    courbe_d_filtered = courbe_d_filter[courbe_d_filter['Région'] == region].reset_index(drop = True)
    conso_r_filtered = conso_r_filter[conso_r_filter['Région'] == region].reset_index(drop = True)  
else :
    courbe_d_filtered = courbe_d_filter.groupby('Date - Heure').agg({'c': 'sum', 't': 'sum', 'r': 'sum'}).reset_index()
    conso_r_filtered = conso_r_filter.groupby('Date - Heure').agg({'consommation brute totale (mw) ':'sum'}).reset_index()
    
courbe_d_filtered['thermosensible'] = conso_r_filtered['consommation brute totale (mw) '] * courbe_d_filtered['t'] / (courbe_d_filtered['t'] + courbe_d_filtered['r'])


fig = px.line(conso_r_filtered, x='Date - Heure', y=['consommation brute totale (mw) '],
              title='Courbe de charge en MW',
              labels={'Date - Heure': 'Date', 'value': 'Consommation (MW)'})
fig.update_traces(name='conso totale')
fig.add_scatter(x=courbe_d_filtered['Date - Heure'], y=courbe_d_filtered['thermosensible'], mode='lines',
                name='conso thermosensible', line=dict(color='blue'))

# Customize fill color
fig.update_traces(fill='tozeroy') 
fig.update_traces(fillcolor='rgba(200, 0, 255)', selector=dict(name='conso thermosensible'))

st.plotly_chart(fig)

st.header("Prévision de l'émission CO2")


############ Partie prévision ############

col21, col22, col23 = st.columns(3)

# Début de l'entraînement
with col21:
    train_start = st.date_input("Début de l'entraînement", datetime.date(2019, 1, 1), min_value = datetime.date(2019, 1, 1))

# Fin de l'entraînement
with col22:
    train_end = st.date_input("Fin de l'entraînement", datetime.date(2019, 10, 1),min_value = train_start, max_value = datetime.date(2023, 12, 31))

# Fin de la prévision
with col23:
    test_end = st.date_input("Fin de la prévision", datetime.date(2019, 12, 31), max_value = datetime.date(2023, 12, 31))


df = pd.read_csv("../Data/carbon_data_2022.csv", sep = ',')

# Modèle à utiliser
modele = st.selectbox(
    'Choisir le Modèle',
    ('LSTM', 'CatBoost','XGBoost')
)

labels = ["Température", "Saisons", "Jours de la semaine", "Vacances", "Jours fériés",
          "Consommation totale", "Conso thermosensible", "Conso régulière", "Lags"]

L = len(labels)

my_dict = {}
my_dict["Saisons"] = ['saison']
my_dict["Jours de la semaine"] = ['week_day']
my_dict["Vacances"] = ['is_holiday']
my_dict["Jours fériés"] = ['is_bank_holiday']
my_dict["Température"] = ['Temp']
my_dict["Consommation totale"] = ['c']
my_dict["Conso régulière"] = ['r']
my_dict["Conso thermosensible"] = ['t']
my_dict["Lags"] = ['Temp1', 'Temp2', 'Temp3', 'Temp4', 'Temp5', 'Temp6', 'Temp7',
                   'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']

# Features à conserver
st.write("Choix des données d'entrée du modèle")

# Create an empty list to store selected labels
selected_labels = []

# Define the number of columns
num_columns = 3

# Create columns for checkbox layout
columns = [st.columns(num_columns) for _ in range(num_columns)]

# Iterate over the labels and create checkboxes
for i, label in enumerate(labels):
    # Determine the column index
    column_index = i % num_columns
    
    # Create the checkbox with a unique key
    checkbox_key = f"{label}_{i}"
    if columns[column_index][i // num_columns].checkbox(label, value=True, key=checkbox_key):
        selected_labels.append(label)

# Use the selected labels to fetch corresponding values from my_dict
selected_features = [my_dict[label] for label in selected_labels]

features = [item for sublist in selected_features for item in sublist]
features.append('Region')
print('selected_features:', features)

if modele == 'LSTM':
    output_df, model = LSTM_predict(df, train_start, train_end, test_end, selected_vars=features)
elif modele == 'CatBoost' :
    output_df, model = Catboost_predict(df, train_start, train_end, test_end, selected_vars=features)
else : 
    output_df, model = Xgboost_predict(df, train_start, train_end, test_end, selected_vars=features)

if region!='Toute la France':
    region_data = output_df[output_df['Region'] == region]
    fig2 = px.line(region_data, x = 'Date', y=['y', 'y_pred'],
              title=f"Emission de CO2 à l'échelle de {region}",
              labels={'Date': 'Date', 'value': 'Emissions (MtCO2/jour)'})
    
else :
    aggregated_data = output_df.groupby(['Date']).apply(pd.DataFrame.sum,skipna=False).drop(columns=['Date','Region']).reset_index()
    fig2 = px.line(aggregated_data, x = 'Date', y=['y', 'y_pred'],
              title="Emission de CO2 à l'échelle de la France",
              labels={'Date': 'Date', 'value': 'Emissions (MtCO2/jour)'})

st.plotly_chart(fig2)

############ Partie évaluation du modèle ############
st.header("Evaluation du modèle de prévision")
# Evaluer le modèle :

is_test = (output_df['Date'] > str(train_end)) * (output_df['Date'] <= str(test_end))

y_test = output_df['y'][is_test]
y_pred = output_df['y_pred'][is_test]

X_train, X_test, _, _, _, _  = encode_normalize(df, train_start, train_end, test_end, features)

# Evaluate 
# Evaluate using Coefficient of Determination (R^2)
r2 = r2_score(y_test, y_pred)
print(f"Coefficient of Determination (R^2): {r2}")
# Evaluate using Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

st.subheader("R² (cefficient de détermination)")
st.latex(r'''
\begin{equation*} 
R^{2} = 1 - \frac{SS_{ee}}{SS_{\text{tot}}}
\end{equation*}
''')
st.markdown("où $SS_{ee}$ est la somme des résidus au carré : $( (y - \hat{y})^2 )$ et $SS_{tot}$ est la somme des carrés : $( (y - mean(y))^2$")
st.write("R² = ", r2)

st.subheader('Mean Absolute Percentage Error')
st.latex(r'''
\begin{equation*}
         MAPE = \frac{100}{n} \sum_{i=1}^{n} \Large\vert \frac{ A_{t} - F_{t}}{A_{t}} \Large\vert
\end{equation*}
''')
st.markdown("où $A_{t}$ est la réalité (actual value), $F_{t}$ est la prédiction (forecast) et $n$ est le nombre d'observations")
st.write("MAPE = ", mape)


############ Partie SHAP ############
st.header("Shapley additive explanation")

# Create a SHAP explainer for your model
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values for your test data
shap_values = explainer.shap_values(X_test)

# Display the SHAP summary plot using Matplotlib
shap.summary_plot(shap_values, X_test)

# Capture the Matplotlib figure
fig, ax = plt.gcf(), plt.gca()

# Render the Matplotlib figure using Streamlit
st.pyplot(fig)


############ Partie simulation ############
st.header("Simulation avec augmentation de la température")
# Augmentation de la température

st.write("On ajoute du bruit sur les températures relevées, tel que :")
st.latex(r'''
\begin{equation*}
        T_{simulated, t} := T_{recorded, t} + \epsilon_{t} \hspace{8mm} \forall t \in \left[t_{0},t_{f}\right]
\end{equation*}
''')

st.write("where")

st.latex(r'''
\begin{equation*}
        \epsilon_{t} \sim \mathcal{N}\left(\frac{T_{inc} \times(t - t_{0})}{t_{f} - t_{0}},\sigma \right)
\end{equation*}
''')


col41, col42 = st.columns(2)
with col41 :
    sigma = st.slider("Ecart-type", min_value = 0.0, max_value = 5.0, value = 1.0)
with col42 :
    temp_incr = st.slider("Augmentation de la température (en °C)", min_value = 0.0, max_value = 5.0, value = 0.5)


train_start = str(train_start)
train_end = str(train_end)
test_end = str(test_end)

def simulate_temperature_recorded(t_recorded, t_inc, t0, tf, sigma=1):
    time_range = np.arange(t0, tf+1)
    epsilon = np.random.normal(loc=(t_inc * (time_range - t0) / (tf - t0)), scale=sigma)
    T_simulated = t_recorded + epsilon
    return time_range, T_simulated

# Define parameters
t_recorded = np.array(df['Temp'][is_test])  # Recorded temperature
t_inc = temp_incr        # Increase in temperature
t0 = 0           # Initial time
tf = len(t_recorded)-1          # Final time
sigma = 1        # Standard deviation for the normal distribution

# Simulate temperature
time_range, T_simulated = simulate_temperature_recorded(t_recorded, t_inc, t0, tf, sigma)

time_range = df['Date'][is_test]
region_range = df['Region'][is_test]

temp_df = pd.DataFrame({'Date' : time_range, 'Region' : region_range, 'Recorded Temperature' : t_recorded, 'Simulated Temperature Rise' : T_simulated})


# Plot the difference in temperature
if region != 'Toute la France':
    # Plot for the region
    region_data3 = temp_df[temp_df['Region'] == region]
    fig3 = px.line(region_data3, x = 'Date', y=['Recorded Temperature', 'Simulated Temperature Rise'],
              title=f"Température à l'échelle de {region}",
              labels={'Date': 'Date', 'value': 'Température (°C)'})
else:
    # Plot for the whole country
    aggregated_data3 = temp_df.groupby(['Date']).apply(pd.DataFrame.mean,skipna=False, numeric_only=True).reset_index()
    fig3 = px.line(aggregated_data3, x = 'Date', y=['Recorded Temperature', 'Simulated Temperature Rise'],
              title="Température à l'échelle de la France",
              labels={'Date': 'Date', 'value': 'Température (°C)'})

st.plotly_chart(fig3)

df2 = df.copy()
df2['Temp'][is_test] = T_simulated

# Model with the new temperature
if modele == 'LSTM':
    output_df2, model2 = LSTM_predict(df2, train_start, train_end, test_end, selected_vars=features)
elif modele == 'CatBoost' :
    output_df2, model2 = Catboost_predict(df2, train_start, train_end, test_end, selected_vars=features)
else : 
    output_df2, model2 = Xgboost_predict(df2, train_start, train_end, test_end, selected_vars=features)


emis_predicted = output_df['y_pred'][is_test]
emis_simulated = output_df2['y_pred'][is_test]
emis_temp_df = pd.DataFrame({'Date' : time_range, 'Region' : region_range, 'Predicted emission' : emis_predicted, 'Predicted emission with temperature rise' : emis_simulated})

if region!='Toute la France':
    region_data4 = emis_temp_df[emis_temp_df['Region'] == region]
    var_emis = np.mean(region_data4['Predicted emission with temperature rise'] - region_data4['Predicted emission'])
    fig4 = px.line(region_data4, x = 'Date', y=['Predicted emission', 'Predicted emission with temperature rise'],
              title=f"Emission de CO2 à l'échelle de {region}",
              labels={'Date': 'Date', 'value': 'Emissions (MtCO2/jour)'})
else :
    aggregated_data4 = emis_temp_df.groupby(['Date']).apply(pd.DataFrame.sum,skipna=False).drop(columns=['Date','Region']).reset_index()
    var_emis = np.mean(aggregated_data4['Predicted emission with temperature rise'] - aggregated_data4['Predicted emission'])
    fig4 = px.line(aggregated_data4, x = 'Date', y=['Predicted emission', 'Predicted emission with temperature rise'],
              title="Emission de CO2 à l'échelle de la France",
              labels={'Date': 'Date', 'value': 'Emissions (MtCO2/jour)'})

st.plotly_chart(fig4)

st.write("Différence moyenne d'émission = ", var_emis)
