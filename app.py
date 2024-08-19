from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import folium
from sqlalchemy import Column, Integer, String
from geoalchemy2 import Geometry
from shapely import wkb
from shapely.geometry import shape
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import pandas as pd

app = Flask(__name__)

# Configuration de la base de données
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:ahmed@localhost:5432/DatabaseLCU'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class GreenSpace(db.Model):
    __tablename__ = 'databaselcu'
    id = Column(Integer, primary_key=True)
    strata = Column(String, name='strata')
    geometry = Column(Geometry('POLYGON'))
    pente_mean = Column(Integer, name='pente_mean')
    pente_variance = Column(Integer, name='pente_variance')
    pente_stdev = Column(Integer, name='pente_stdev')
    altitude_mean = Column(Integer, name='altitude_mean')
    altitude_max = Column(Integer, name='altitude_max')
    exposition_mean = Column(Integer, name='exposition_mean')
    exposition_variance = Column(Integer, name='exposition_variance')
    exposition_stdev = Column(Integer, name='exposition_stdev')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/acceuil')
def accueil():
    return render_template('accueil.html')

@app.route('/eda')
def eda():
    # Calcul des statistiques globales
    stats = {
        'total_points': db.session.query(db.func.count(GreenSpace.id)).scalar(),
        'average_pente': db.session.query(db.func.avg(GreenSpace.pente_mean)).scalar(),
        'variance_pente': db.session.query(db.func.avg(GreenSpace.pente_variance)).scalar(),
        'stdev_pente': db.session.query(db.func.avg(GreenSpace.pente_stdev)).scalar(),
        'average_altitude': db.session.query(db.func.avg(GreenSpace.altitude_mean)).scalar(),
        'max_altitude': db.session.query(db.func.max(GreenSpace.altitude_max)).scalar(),
        'average_exposition': db.session.query(db.func.avg(GreenSpace.exposition_mean)).scalar(),
        'variance_exposition': db.session.query(db.func.avg(GreenSpace.exposition_variance)).scalar(),
        'stdev_exposition': db.session.query(db.func.avg(GreenSpace.exposition_stdev)).scalar(),
    }

    return render_template('eda.html', stats=stats)

@app.route('/eda/strata_altitude')
def eda_strata_altitude():
    data = db.session.query(GreenSpace.strata, GreenSpace.altitude_mean).limit(2000).all()
    df = pd.DataFrame(data, columns=['strata', 'altitude_mean'])

    import plotly.express as px
    fig = px.bar(df, x='strata', y='altitude_mean', color='strata',
                 title="Répartition des Strates par Altitude",
                 labels={'altitude_mean': 'Altitude Moyenne'})
    graph_html = fig.to_html(full_html=False)

    return render_template('eda.html', strata_altitude_graph_html=graph_html)

@app.route('/eda/exposition_altitude')
def eda_exposition_altitude():
    data = db.session.query(GreenSpace.exposition_mean, GreenSpace.altitude_mean).all()
    df = pd.DataFrame(data, columns=['exposition_mean', 'altitude_mean'])

    fig = px.scatter(df, x='altitude_mean', y='exposition_mean',
                     title="Exposition vs Altitude",
                     labels={'altitude_mean': 'Altitude Moyenne', 'exposition_mean': 'Exposition Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', exposition_altitude_graph_html=graph_html)

@app.route('/eda/boxplot_exposition')
def eda_boxplot_exposition():
    data = db.session.query(GreenSpace.strata, GreenSpace.exposition_mean).all()
    df = pd.DataFrame(data, columns=['strata', 'exposition_mean'])

    fig = px.box(df, x='strata', y='exposition_mean',
                 title="Distribution des Expositions par Strate",
                 labels={'exposition_mean': 'Exposition Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', boxplot_exposition_graph_html=graph_html)

@app.route('/eda/histogramme_altitude')
def eda_histogramme_altitude():
    data = db.session.query(GreenSpace.strata, GreenSpace.altitude_mean).all()
    df = pd.DataFrame(data, columns=['strata', 'altitude_mean'])

    fig = px.histogram(df, x='altitude_mean', color='strata',
                       title="Histogramme des Altitudes par Strate",
                       labels={'altitude_mean': 'Altitude Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', histogramme_altitude_graph_html=graph_html)

@app.route('/eda/piechart_strata_exposition')
def eda_piechart_strata_exposition():
    data = db.session.query(GreenSpace.strata, GreenSpace.exposition_mean).all()
    df = pd.DataFrame(data, columns=['strata', 'exposition_mean'])

    # Créez des catégories d'exposition (par exemple, Nord, Sud, Est, Ouest)
    df['exposition_group'] = pd.cut(df['exposition_mean'], bins=[0, 90, 180, 270, 360], labels=["Nord", "Est", "Sud", "Ouest"])

    fig = px.pie(df, names='exposition_group', color='strata', title="Répartition des Types de Sol par Exposition")
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', piechart_strata_exposition_graph_html=graph_html)

@app.route('/eda/heatmap_altitude')
def eda_heatmap_altitude():
    data = db.session.query(GreenSpace.altitude_mean, GreenSpace.geometry).all()
    coords = [(shape(wkb.loads(bytes(space.geometry.data))).centroid.y, shape(wkb.loads(bytes(space.geometry.data))).centroid.x) for space in data]
    altitudes = [space.altitude_mean for space in data]

    m = folium.Map(location=[31.7917, -7.0926], zoom_start=6)
    HeatMap(list(zip([coord[0] for coord in coords], [coord[1] for coord in coords], altitudes)), radius=10).add_to(m)

    map_html = m._repr_html_()
    return render_template('eda.html', heatmap_altitude_map_html=map_html)

@app.route('/eda/histogramme_exposition')
def eda_histogramme_exposition():
    data = db.session.query(GreenSpace.exposition_mean).all()
    df = pd.DataFrame(data, columns=['exposition_mean'])

    fig = px.histogram(df, x='exposition_mean',
                       title="Histogramme de la Distribution des Expositions",
                       labels={'exposition_mean': 'Exposition Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', histogramme_exposition_graph_html=graph_html)

@app.route('/visualiser')
def visualiser():
    m = folium.Map(location=[31.7917, -7.0926], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)
    green_spaces = GreenSpace.query.limit(10000).all()
    for space in green_spaces:
        geom = shape(wkb.loads(bytes(space.geometry.data)))
        popup_content = (
            f"ID: {space.id}<br>"
            f"Strata: {space.strata}<br>"
            f"Pente Moyenne: {space.pente_mean}<br>"
            f"Variance de la Pente: {space.pente_variance}<br>"
            f"Écart-type de la Pente: {space.pente_stdev}<br>"
            f"Altitude Moyenne: {space.altitude_mean}<br>"
            f"Altitude Maximale: {space.altitude_max}<br>"
            f"Exposition Moyenne: {space.exposition_mean}<br>"
            f"Variance de l'Exposition: {space.exposition_variance}<br>"
            f"Écart-type de l'Exposition: {space.exposition_stdev}"
        )
        folium.GeoJson(geom, name=space.strata, popup=popup_content).add_to(marker_cluster)

    map_html = m._repr_html_()
    return render_template('visualiser.html', map_html=map_html)

@app.route('/pred')
def pred():
    return render_template('pred.html')

if __name__ == '__main__':
    app.run(debug=True)
