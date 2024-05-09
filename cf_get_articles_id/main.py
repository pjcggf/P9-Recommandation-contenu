"""Fonction de predictions des 5 articles grâce au Collaborative Filtering."""
import pickle
import os
from io import BytesIO
import flask
import functions_framework
from google.cloud import storage

####
# A définir seulement en test
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../gcs_reader_cred.json'
#####
client_gcs = storage.Client(project='p9-reco-contenu')
bucket = client_gcs.get_bucket('data-p9-reco')

model_raw = BytesIO(bucket.get_blob(
    'model_collab_filter.pkl').download_as_bytes())
model_collab_filtering = pickle.load(model_raw)


sparse_matrix_raw = BytesIO(bucket.get_blob(
    'sparse_user_item.pkl').download_as_bytes())
sparse_matrix = pickle.load(sparse_matrix_raw)

@functions_framework.http
def get_articles_id(request: flask.Request) -> flask.typing.ResponseReturnValue:
    '''
    Renvoie 5 articles suggérés pour l'utilisateur.
    Args:
        user_id: int -> Id de l'utilisateur pour la prédiction à réaliser.
        n: int -> Nombre de recommandations renvoyées.
    '''
    user_id = request.args.get('user_id')
    if not user_id:
        raise ValueError("Le user_id doit être spécifié")
    user_id = int(user_id)

    try:
        n = int(request.args.get('n'))
    except TypeError:
        n=5

    try:
        recommended = model_collab_filtering.recommend(user_id, sparse_matrix, N=n)
    except IndexError as e:
        raise IndexError(f"""Le user_id {user_id} n'existe pas.
                         Réssayer avec un user_id valide""") from e
    #  Retourne les 5 meilleures recommandations
    return {f'{i}': f"{v}" for i, v in recommended}
