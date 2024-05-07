"""Fonction pour récupérer la liste des id des user"""
import flask
import functions_framework
from google.cloud import bigquery

PROJECT = "p9-reco-contenu"
DATASET = "p9-reco"
TABLE = "train"

client = bigquery.Client(project=PROJECT)


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

    query = f"""
        SELECT user_id
        FROM {PROJECT}.{DATASET}.{TABLE}
        ORDER BY RAND()
        LIMIT {nb}
        """

    query_job = client.query(query)
    result = query_job.result()
    return result
