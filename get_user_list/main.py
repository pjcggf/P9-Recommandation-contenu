"""Fonction pour récupérer la liste des id des user"""
from io import BytesIO
import random
import pandas as pd
import flask
import functions_framework
from google.cloud import storage

client_gcs = storage.Client(project='p9-reco-contenu')
bucket = client_gcs.get_bucket('data-p9-reco')


def get_list():
    '''Précharge le jeu de données et récupère la liste comme variable
    d'instance (réduit le temps de latence pour les démarrages à froid).'''
    train_raw = BytesIO(bucket.get_blob(
        'train.feather').download_as_bytes())
    liste = pd.read_feather(train_raw).user_id.unique().tolist()

    return liste


user_list = get_list()


@functions_framework.http
def get_user_list(request: flask.Request) -> flask.typing.ResponseReturnValue:
    '''
    Renvoie la liste des ids de users contenue dant train.feather
    nb: int : nombre de user_id souhaité
    '''
    try:
        nb = int(request.args.get('nb'))
    except TypeError:
        nb = 100

    res = random.sample(user_list, nb)
    return res
