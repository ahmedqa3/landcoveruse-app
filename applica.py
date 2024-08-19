from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import folium
from sqlalchemy import Column, Integer, String
from geoalchemy2 import Geometry
from shapely import wkb
from shapely.geometry import shape
from folium import GeoJson
from folium.plugins import MarkerCluster

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

@app.route('/visualiser')
def index():
    # Créer une carte Folium centrée sur le Maroc
    m = folium.Map(location=[31.7917, -7.0926], zoom_start=6)

    # Créer un cluster de marqueurs
    marker_cluster = MarkerCluster().add_to(m)

    # Ajouter les polygones à la carte
    green_spaces = GreenSpace.query.limit(10000).all()
    for space in green_spaces:
        # Convertir WKB en géométrie Shapely
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
        # Ajouter le polygone au cluster
        folium.GeoJson(geom, name=space.strata, popup=popup_content).add_to(marker_cluster)

    # Générer le HTML de la carte
    map_html = m._repr_html_()
    return render_template('applica.html', map_html=map_html)

@app.route('/green_space/<int:space_id>')
def get_green_space(space_id):
    space = GreenSpace.query.get_or_404(space_id)
    space_data = {
        'id': space.id,
        'strata': space.strata,
        'geometry': shape(wkb.loads(bytes(space.geometry.data))).__geo_interface__,
        'pente_mean': space.pente_mean,
        'pente_variance': space.pente_variance,
        'pente_stdev': space.pente_stdev,
        'altitude_mean': space.altitude_mean,
        'altitude_max': space.altitude_max,
        'exposition_mean': space.exposition_mean,
        'exposition_variance': space.exposition_variance,
        'exposition_stdev': space.exposition_stdev,
    }
    return jsonify(space_data)

@app.route('/data')
def data():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 3000, type=int)  # Nombre d'éléments par page
    strata_filter = request.args.get('strata', 'all')

    query = GreenSpace.query

    if strata_filter != 'all':
        query = query.filter(GreenSpace.strata == strata_filter)

    # Pagination des données
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    features = []
    for space in pagination.items:
        geom = shape(wkb.loads(bytes(space.geometry.data)))
        features.append({
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': {
                'id': space.id,
                'strata': space.strata,
                'pente_mean': space.pente_mean,
                'pente_variance': space.pente_variance,
                'pente_stdev': space.pente_stdev,
                'altitude_mean': space.altitude_mean,
                'altitude_max': space.altitude_max,
                'exposition_mean': space.exposition_mean,
                'exposition_variance': space.exposition_variance,
                'exposition_stdev': space.exposition_stdev,
            }
        })
    return jsonify({
        'type': 'FeatureCollection',
        'features': features,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'total': pagination.total
    })


if __name__ == '__main__':
    app.run(debug=True)
