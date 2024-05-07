"""Fonction pour récupérer la liste des id des user"""
from io import BytesIO
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
    list = pd.read_feather(BytesIO(train_raw)).user_id.unique()

    return list

user_list = get_list()

@functions_framework.http
def get_user_list(request: flask.Request) -> flask.typing.ResponseReturnValue:
    '''
    Renvoie la liste des ids de users contenue dant train.feather
    '''
    return user_list
