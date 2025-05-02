
from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

# ----------------------------
# Cargar y preparar datos
# ----------------------------
sales_df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')
sales_df.drop(['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'ORDERNUMBER'], axis=1, inplace=True)
sales_df['COUNTRY'] = sales_df['COUNTRY'].str.strip()
sales_df = pd.get_dummies(sales_df, columns=['COUNTRY', 'PRODUCTLINE', 'DEALSIZE'])
sales_df['PRODUCTCODE'] = pd.Categorical(sales_df['PRODUCTCODE']).codes
sales_df.drop("ORDERDATE", axis=1, inplace=True)
sales_df['ORDERLINENUMBER'] = sales_df['ORDERLINENUMBER'].astype(float)

scaler = StandardScaler()
sales_df_scaled = scaler.fit_transform(sales_df.select_dtypes(include='number'))

# KMeans y PCA
kmeans = KMeans(n_clusters=5, n_init=10)
labels = kmeans.fit_predict(sales_df_scaled)
pca = PCA(n_components=3)
principal_components = pca.fit_transform(sales_df_scaled)
pca_df = pd.DataFrame(principal_components, columns=['pca1', 'pca2', 'pca3'])
pca_df['cluster'] = labels

# Gráfico 3D
fig3d = px.scatter_3d(pca_df, x='pca1', y='pca2', z='pca3', color=pca_df['cluster'].astype(str))
pio.write_html(fig3d, file='static/grafico3d.html', auto_open=False)

# Gráfico de barras (COUNTRY)
def create_bar_plot():
    df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')
    bar_fig = px.bar(x=df['COUNTRY'].value_counts().index, y=df['COUNTRY'].value_counts(), title='Ventas por País')
    pio.write_html(bar_fig, file='static/bar_plot.html', auto_open=False)

# Matriz de correlación
def create_corr_plot():
    corr = pd.DataFrame(sales_df_scaled, columns=sales_df.select_dtypes(include='number').columns).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de Correlación")
    plt.tight_layout()
    plt.savefig("static/corr_plot.png")
    plt.close()

# Crear visualizaciones al iniciar
create_bar_plot()
create_corr_plot()

@app.route('/')
def index():
    table_html = sales_df.head(10).to_html(classes='table table-striped', border=0)
    return render_template("index.html", table=table_html)

if __name__ == '__main__':
    app.run(debug=True)
