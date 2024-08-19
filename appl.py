from flask import Flask, render_template, jsonify, request, session
from flask_sqlalchemy import SQLAlchemy
import folium
from sqlalchemy import Column, Integer, String
from geoalchemy2 import Geometry
from shapely import wkb
from shapely.geometry import shape
from folium import GeoJson
import plotly.express as px
import pandas as pd
import folium.plugins
from folium.plugins import MarkerCluster, HeatMap
from sklearn.svm import SVC
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from flask import render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os


app = Flask(__name__)

# Configuration de la base de données
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:ahmed@localhost:5432/DatabaseLCU')
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
    return render_template('c.html')

@app.route('/stat')
def stat():
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

@app.route('/eda/exposition_altitude')
def exposition_altitude():
    data = db.session.query(GreenSpace.exposition_mean, GreenSpace.altitude_mean).all()
    df = pd.DataFrame(data, columns=['exposition_mean', 'altitude_mean'])

    fig = px.scatter(df, x='altitude_mean', y='exposition_mean',
                     labels={'altitude_mean': 'Altitude Moyenne', 'exposition_mean': 'Exposition Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', exposition_altitude_graph_html=graph_html)




@app.route('/eda/boxplot_exposition')
def boxplot_exposition():
    data = db.session.query(GreenSpace.strata, GreenSpace.exposition_mean).all()
    df = pd.DataFrame(data, columns=['strata', 'exposition_mean'])

    fig = px.box(df, x='strata', y='exposition_mean',
                 title="Distribution des Expositions par Strate",
                 labels={'exposition_mean': 'Exposition Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', boxplot_exposition_graph_html=graph_html)


@app.route('/eda/histogramme_altitude')
def histogramme_altitude():
    data = db.session.query(GreenSpace.strata, GreenSpace.altitude_mean).all()
    df = pd.DataFrame(data, columns=['strata', 'altitude_mean'])

    fig = px.histogram(df, x='altitude_mean', color='strata',
                       title="Histogramme des Altitudes par Strate",
                       labels={'altitude_mean': 'Altitude Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', histogramme_altitude_graph_html=graph_html)


@app.route('/eda/piechart_strata_exposition')
def piechart_strata_exposition():
    data = db.session.query(GreenSpace.strata, GreenSpace.exposition_mean).all()
    df = pd.DataFrame(data, columns=['strata', 'exposition_mean'])

    df['exposition_group'] = pd.cut(df['exposition_mean'], bins=[0, 90, 180, 270, 360], labels=["Nord", "Est", "Sud", "Ouest"])

    fig = px.pie(df, names='exposition_group', color='strata')
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', piechart_strata_exposition_graph_html=graph_html)


@app.route('/eda/heatmap_altitude')
def heatmap_altitude():
    page = int(request.args.get('page', 1))  # Obtenez la page actuelle, défaut à 1
    per_page = 30000  # Nombre d'éléments par page
    offset = (page - 1) * per_page  # Calculer l'offset pour la pagination

    # Récupérer les données pour la page actuelle
    data = db.session.query(GreenSpace.altitude_mean, GreenSpace.geometry).offset(offset).limit(per_page).all()
    coords = [(shape(wkb.loads(bytes(space.geometry.data))).centroid.y, shape(wkb.loads(bytes(space.geometry.data))).centroid.x) for space in data]
    altitudes = [space.altitude_mean for space in data]

    m = folium.Map(location=[31.7917, -7.0926], zoom_start=6)
    HeatMap(list(zip([coord[0] for coord in coords], [coord[1] for coord in coords], altitudes)), radius=10).add_to(m)

    map_html = m._repr_html_()

    # Calculer le nombre total de pages
    total_items = db.session.query(GreenSpace).count()
    total_pages = (total_items + per_page - 1) // per_page

    return render_template('eda.html', heatmap_altitude_map_html=map_html, current_page=page, total_pages=total_pages)


@app.route('/eda/histogramme_exposition')
def histogramme_exposition():
    data = db.session.query(GreenSpace.exposition_mean).all()
    df = pd.DataFrame(data, columns=['exposition_mean'])

    fig = px.histogram(df, x='exposition_mean',
                       title="Histogramme de la Distribution des Expositions",
                       labels={'exposition_mean': 'Exposition Moyenne'})
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', histogramme_exposition_graph_html=graph_html)




@app.route('/eda/scatter_plot', methods=['GET', 'POST'])
def scatter_plot():
    if request.method == 'POST':
        x_var = request.form['x_variable']
        y_var = request.form['y_variable']

        # Récupération des données de la base de données
        data = db.session.query(getattr(GreenSpace, x_var), getattr(GreenSpace, y_var)).all()

        # Conversion des tuples en deux listes séparées
        x_data, y_data = zip(*data)

        # Création du DataFrame avec les listes unidimensionnelles
        df = pd.DataFrame({x_var: x_data, y_var: y_data})

        # Création du scatter plot avec Plotly Express
        fig = px.scatter(df, x=x_var, y=y_var,
                         labels={x_var: x_var.replace('_', ' ').title(), 
                                 y_var: y_var.replace('_', ' ').title()},
                         title=f'Scatter Plot: {x_var} vs {y_var}')
        graph_html = fig.to_html(full_html=False)
        return render_template('eda.html', scatter_plot_html=graph_html)

    available_vars = ['pente_mean', 'pente_variance', 'pente_stdev', 'altitude_mean', 'altitude_max',
                      'exposition_mean', 'exposition_variance', 'exposition_stdev']
    return render_template('scatter.html', available_vars=available_vars)


@app.route('/eda/correlation_matrix')
def correlation_matrix():
    data = db.session.query(
        GreenSpace.pente_mean,
        GreenSpace.pente_variance,
        GreenSpace.pente_stdev,
        GreenSpace.altitude_mean,
        GreenSpace.altitude_max,
        GreenSpace.exposition_mean,
        GreenSpace.exposition_variance,
        GreenSpace.exposition_stdev
    ).all()
    df = pd.DataFrame(data, columns=[
        'Pente Moyenne',
        'Variance Pente',
        'Écart-type Pente',
        'Altitude Moyenne',
        'Altitude Maximale',
        'Exposition Moyenne',
        'Variance Exposition',
        'Écart-type Exposition'
    ])
    correlation_matrix = df.corr()

    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                    labels=dict(color="Corrélation"))
    graph_html = fig.to_html(full_html=False)
    return render_template('eda.html', correlation_matrix_graph_html=graph_html)

@app.route('/eda/correlation_custom', methods=['GET', 'POST'])
def correlation_custom():
    if request.method == 'POST':
        variables = request.form.getlist('variables')
        data = db.session.query(*(getattr(GreenSpace, var) for var in variables)).all()
        df = pd.DataFrame(data, columns=variables)
        correlation_matrix = df.corr()

        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                        labels=dict(color="Corrélation"))
        graph_html = fig.to_html(full_html=False)
        return render_template('eda.html', correlation_matrix_graph_html=graph_html)

    # Variables disponibles pour la sélection
    available_vars = ['pente_mean', 'pente_variance', 'pente_stdev', 'altitude_mean', 'altitude_max',
                      'exposition_mean', 'exposition_variance', 'exposition_stdev']
    return render_template('cor_cus.html', available_vars=available_vars)


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
    return render_template('applica.html', map_html=map_html)

@app.route('/pred')
def pred():
    return render_template('models.html')




@app.route('/pred/tree_cover_change_detection', methods=['GET', 'POST'])
def pred_tree_cover_change_detection():
    if request.method == 'POST':
        x_vars = ['altitude_mean', 'pente_mean', 'exposition_mean']
        data = db.session.query(*[getattr(GreenSpace, var) for var in x_vars], GreenSpace.strata).limit(50000).all()
        df = pd.DataFrame(data, columns=x_vars + ['strata'])
        X = df[x_vars]
        y = df['strata']

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        df['predictions'] = model.predict(X)

        # Évaluation des performances
        report = classification_report(y, df['predictions'], output_dict=True)
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        # Visualisation des prédictions
        fig = px.scatter(df, x=x_vars[0], y='strata', color='predictions')
        graph_html = fig.to_html(full_html=False)
        
        return render_template(
            'pred.html',
            model_summary="Détection du Changement de Couverture Arborée",
            graph_html=graph_html,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
    
    return render_template('model_summary.html', route='pred_tree_cover_change_detection')


@app.route('/pred/vegetation_density_estimation', methods=['GET', 'POST'])
def pred_vegetation_density_estimation():
    if request.method == 'POST':
        x_vars = ['altitude_mean', 'pente_mean', 'exposition_mean']
        data = db.session.query(*[getattr(GreenSpace, var) for var in x_vars], GreenSpace.strata).limit(100000).all()
        df = pd.DataFrame(data, columns=x_vars + ['strata'])

        label_encoder = LabelEncoder()
        df['strata_encoded'] = label_encoder.fit_transform(df['strata'])
        
        X = df[x_vars]
        y = df['strata_encoded']
        
        model = LinearRegression()
        model.fit(X, y)
        df['predictions'] = model.predict(X)

        # Calculer R2, MSE, et MAE
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        r2 = r2_score(y, df['predictions'])
        mse = mean_squared_error(y, df['predictions'])
        mae = mean_absolute_error(y, df['predictions'])
        
        fig = px.scatter(df, x=x_vars[0], y='predictions', trendline="ols")
        graph_html = fig.to_html(full_html=False)
        
        return render_template(
            'predc.html',
            model_summary="Estimation de la Densité Végétale",
            graph_html=graph_html,
            r2=r2,
            mse=mse,
            mae=mae
        )
    
    return render_template('model_summary.html', route='pred_vegetation_density_estimation')




@app.route('/pred/land_cover_topography_analysis', methods=['GET', 'POST'])
def pred_land_cover_topography_analysis():
    if request.method == 'POST':
        x_vars = ['altitude_mean', 'pente_mean', 'exposition_mean']
        data = db.session.query(*[getattr(GreenSpace, var) for var in x_vars], GreenSpace.strata).all()
        df = pd.DataFrame(data, columns=x_vars + ['strata'])

        # Encoder la variable cible (strata) avec LabelEncoder
        label_encoder = LabelEncoder()
        df['strata_encoded'] = label_encoder.fit_transform(df['strata'])

        X = df[x_vars]
        y = df['strata_encoded']  # Utiliser les valeurs numériques encodées

        model = LinearRegression()
        model.fit(X, y)
        df['predictions'] = model.predict(X)

        # Calculer R2, MSE, et MAE
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        r2 = r2_score(y, df['predictions'])
        mse = mean_squared_error(y, df['predictions'])
        mae = mean_absolute_error(y, df['predictions'])

        # Créer un graphique pour les variables
        fig = px.scatter_matrix(df, dimensions=x_vars, color='strata')
        graph_html = fig.to_html(full_html=False)

        return render_template(
            'predc.html',
            model_summary="Analyse des Corrélations entre Couverture du Sol et Topographie",
            graph_html=graph_html,
            r2=r2,
            mse=mse,
            mae=mae
        )
    
    return render_template('model_summary.html', route='pred_land_cover_topography_analysis')






@app.route('/pred/tree_cover_change', methods=['GET', 'POST'])
def pred_tree_cover_change():
    if request.method == 'POST':
        # Variables explicatives (features)
        x_vars = ['altitude_mean', 'exposition_mean']  # Exclure 'pente_mean'

        # Récupérer les données de la base de données
        data = db.session.query(*[getattr(GreenSpace, var) for var in x_vars], GreenSpace.strata).all()

        # Convertir les données en DataFrame pandas
        df = pd.DataFrame(data, columns=x_vars + ['strata'])

        # Filtrer les données pour ne garder que les classes pertinentes
        relevant_strata = ['Tree cover gain', 'Tree cover loss, not fire', 
                           'Wetland tree cover gain', 'Wetland tree cover loss, not fire']
        df = df[df['strata'].isin(relevant_strata)]

        # Encoder la variable cible (strata) avec LabelEncoder
        label_encoder = LabelEncoder()
        df['strata_encoded'] = label_encoder.fit_transform(df['strata'])

        # Vérifier la taille de chaque classe
        class_counts = df['strata'].value_counts()
        min_count = class_counts.min()
        
        # Échantillonnage équilibré
        df_balanced = pd.DataFrame()
        for strata in relevant_strata:
            df_strata = df[df['strata'] == strata]
            if len(df_strata) > 0:
                # Utiliser min_count ou la taille réelle si min_count est trop grand
                n_samples = min(len(df_strata), min_count)
                df_sampled = resample(df_strata, replace=True, n_samples=n_samples, random_state=42)
                df_balanced = pd.concat([df_balanced, df_sampled])
        
        # Séparer les features (X) et la variable cible (y)
        X = df_balanced[x_vars]
        y = df_balanced['strata_encoded']

        # Standardiser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entraînement du modèle de régression logistique
        model = LogisticRegression(max_iter=1000)  # Augmenter à 1000 itérations
        model.fit(X_scaled, y)

        # Prédictions du modèle
        y_pred = model.predict(X_scaled)
        df_balanced['predictions'] = y_pred
        df_balanced['predicted_strata'] = label_encoder.inverse_transform(y_pred)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        # Calcul des métriques
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        # Créer la figure Plotly Express
        fig = px.scatter(
            df_balanced, 
            x=x_vars[0], 
            y=x_vars[1], 
            color='predicted_strata',
            labels={x_vars[0]: x_vars[0], x_vars[1]: x_vars[1]},
            color_discrete_map={
                'Tree cover gain': 'green', 
                'Tree cover loss, not fire': 'red', 
                'Wetland tree cover gain': 'blue', 
                'Wetland tree cover loss, not fire': 'purple'
            }
        )

        # Convertir la figure en HTML
        graph_html = fig.to_html(full_html=False)

        # Renvoyer le graphique et les métriques à la page HTML
        return render_template('predc.html', 
                               model_summary="Prédictions de Changement de Couverture",
                               graph_html=graph_html,
                               r2=r2,
                               mse=mse,
                               mae=mae)

    return render_template('model_summary.html', route='pred_tree_cover_change')


















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
