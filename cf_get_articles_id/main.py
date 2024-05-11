"""Fonction de predictions des 5 articles grâce au Collaborative Filtering."""
import pickle
import os
import warnings
from io import BytesIO
import flask
import functions_framework
from google.cloud import storage, bigquery

####
# A définir seulement en test
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../gcs_reader_cred.json'
#####

PROJECT = 'p9-reco-contenu'
client_gcs = storage.Client(project=PROJECT)
bucket = client_gcs.get_bucket('data-p9-reco')

client_bigquery = bigquery.Client(project=PROJECT)

model_raw = BytesIO(bucket.get_blob(
    'model_collab_filter.pkl').download_as_bytes())
model_collab_filtering = pickle.load(model_raw)


sparse_matrix_raw = BytesIO(bucket.get_blob(
    'sparse_user_item.pkl').download_as_bytes())
sparse_matrix = pickle.load(sparse_matrix_raw)


def user_exist(user_id):
    """Vérifie si le user_id existe dans la base"""
    query = f"""
            SELECT
                DISTINCT(user_id)
            FROM
                `{PROJECT}.p9_reco.train`
            WHERE user_id = {user_id}
            """
    query_job = client_bigquery.query(query)
    result = query_job.result().to_dataframe()

    return result


@functions_framework.http
def get_articles_id(request: flask.Request) -> flask.typing.ResponseReturnValue:
    '''
    Renvoie 5 articles suggérés pour l'utilisateur.
    Args:
        user_id: int -> Id de l'utilisateur pour la prédiction à réaliser.
        n: int -> Nombre de recommandations renvoyées.
    '''
    try:
        user_id = request.args.get('user_id')
        user_id = int(user_id)
    except (TypeError, ValueError):
        warnings.warn(
            'Pas de user_id précisé ou invalide, prédiction impossible.')
        res = {0: 'Pas de user_id précisé ou invalide, prédiction impossible.'}
        return res, 400

    if user_exist(user_id).empty:
        warnings.warn(
            'User_id inconnu, prédiction impossible.')
        res = {0: 'User_id inconnu, prédiction impossible.'}
        return res, 400

    try:
        n = int(request.args.get('n'))
    except TypeError:
        n = 5

    recommended = model_collab_filtering.recommend(user_id, sparse_matrix, N=n)

    #  Retourne les 5 meilleures recommandations
    return {f'{i}': f"{v}" for i, v in recommended}
