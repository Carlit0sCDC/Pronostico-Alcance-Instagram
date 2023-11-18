#importamos las librerias necesarias
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_white'

#leemos el archivo csv
data = pd.read_csv('Instagram-Reach.csv')

#antes de continuar convertimos la columa 'Date' en datetime para que no de errores
data['Date'] = pd.to_datetime(data['Date'])

import statsmodels.api as sm
import warnings

p, d, q = 8, 1, 2

modelo = sm.tsa.statespace.SARIMAX(data['Instagram reach'],
                                   order = (p,d,q), seasonal_order = (p,d,q,12))
modelo = modelo.fit()
print(modelo.summary())

#ahora hagamos las predicciones usando el modelo y veamos el pronostico del alcance
predicciones = modelo.predict(len(data), len(data) + 100)

trace_train = go.Scatter(x = data.index, y = data['Instagram reach'],
                         mode = 'lines',
                         name = 'Data de entrenamiento')
trace_pred = go.Scatter(x = predicciones.index, y = predicciones,
                        mode = 'lines',
                        name = 'Predicciones')
layout = go.Layout(title = 'Prediccion del alcance de instagram con Series temporales',
                   xaxis_title = 'Date',
                   yaxis_title = 'Instagram reach')

fig = go.Figure(data = [trace_train, trace_pred], layout = layout)
fig.show()