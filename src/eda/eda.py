import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessing.load_dataset import load_dataset
from src.preprocessing.preprocessing import preprocessing

def eda(data):
    if not os.path.exists('./files/figs/'):
        os.makedirs('./files/figs/')
    output_path='./files/figs/'
    top10_products=data.groupby(['product_id','product_title'],as_index=False)['customer_id'].count().sort_values(by='customer_id',ascending=False).head(10)
    top10_products
    top10_products_plot=data['product_id'].value_counts().sort_values(ascending=False).head(10)
    top10_products_plot
    fig,ax=plt.subplots(figsize=(10,10))
    top10_products_plot.plot(kind='bar',ax=ax,rot=45)
    ax.set_title('Top reviewed products')
    for container in ax.containers:
        ax.bar_label(container)
    fig.savefig(output_path+'top10_products.png')
    # Agregamos el grafico de los 10 productos con más reviews
    fig1,ax1=plt.subplots(figsize=(10,10))
    top10_products_sentiment=data.groupby(['product_id','sentiment'],as_index=False)['customer_id'].count().sort_values(by='customer_id',ascending=False)
    top10_products_sentiment_merged=top10_products_sentiment.merge(top10_products,on='product_id',how='inner')
    top10_products_sentiment_merged=top10_products_sentiment_merged[['product_id','sentiment','customer_id_x']]
    sns.barplot(data=top10_products_sentiment_merged,x='product_id',y='customer_id_x',hue='sentiment',ax=ax1)
    # Agregar etiquetas sobre las barras
    for container in ax1.containers:  # Iterar sobre las barras
        ax1.bar_label(container, fmt='%.0f', label_type='edge', padding=3)
    # Agregar etiquetas a los ejes y título
    ax1.set_title('Análisis de Sentimientos - Top 10 Productos')  # Título
    ax1.set_xlabel('ID del Producto')  # Etiqueta del eje X
    ax1.set_ylabel('Cantidad de Clientes')  # Etiqueta del eje Y
    plt.xticks(rotation=45)
    fig1.savefig(output_path+'top10_products_reviews.png')

    # Agregamos el grafico la distribución del sentimiento si la compra está verificada o no
    fig2,ax2=plt.subplots(figsize=(10,10))
    sentiment_verified=data.groupby(['sentiment','verified_purchase'],as_index=False)['product_id'].count().sort_values(by='product_id',ascending=False)
    sns.barplot(data=sentiment_verified,x='sentiment',y='product_id',hue='verified_purchase',ax=ax2)
    # Agregar etiquetas sobre las barras
    for container in ax2.containers:  # Iterar sobre las barras
        ax2.bar_label(container, fmt='%.0f', label_type='edge', padding=3)
    # Agregar etiquetas a los ejes y título
    plt.xticks(rotation=45)
    fig2.savefig(output_path+'sentiment_verified.png')
    # Agregamos el grafico la distribución del sentimiento si la compra está soportada con vine para probar la efectividad de este servicio de amazon
    fig3,ax3=plt.subplots(figsize=(20,10))
    sentiment_vine=data.groupby(['sentiment','vine'],as_index=False)['product_id'].count().sort_values(by='product_id',ascending=False)
    sns.barplot(data=sentiment_vine,x='sentiment',y='product_id',hue='vine',ax=ax3)
    # Agregar etiquetas sobre las barras
    for container in ax3.containers:  # Iterar sobre las barras
        ax3.bar_label(container, fmt='%.0f', label_type='edge', padding=3)
    # Agregar etiquetas a los ejes y título
    plt.xticks(rotation=45)
    fig3.savefig(output_path+'sentiment_vine.png')
    
    # Influencia de vine en la calificación
    fig4,ax4=plt.subplots(figsize=(7,10))
    rating=data.groupby(['vine'],as_index=False)['star_rating'].mean()
    ax4.bar(rating['vine'],rating['star_rating'])
    ax4.set_title('Distribución de las calificaciones')
    for container in ax4.containers:  # Iterar sobre las barras
        ax4.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
    fig4.savefig(output_path+'vine_influence.png')
    
    # Influencia de las compras verificadas en la calificación
    fig5,ax5=plt.subplots(figsize=(7,10))
    rating=data.groupby(['verified_purchase'],as_index=False)['star_rating'].mean()
    ax5.bar(rating['verified_purchase'],rating['star_rating'])
    ax5.set_title('Distribución de las calificaciones')
    for container in ax5.containers:  # Iterar sobre las barras
        ax5.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
    fig5.savefig(output_path+'v_purchase_influence.png')
    
    
    vine_sales=data[data['verified_purchase']=='Y']
    fig6,ax6=plt.subplots(figsize=(5,10))
    count_vine=vine_sales['vine'].value_counts()
    count_vine.plot(kind='bar',ax=ax6)
    ax6.set_title('Vine impact for purchased items')
    for container in ax6.containers:  # Iterar sobre las barras
        ax6.bar_label(container, fmt='%.0f', label_type='edge', padding=3)
    fig6.savefig(output_path+'vine_impact.png')