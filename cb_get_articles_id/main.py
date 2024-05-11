"""Prédiction des 5 articles les plus proches des derniers articles lu
    par le user."""
from io import BytesIO
import warnings
import functions_framework
from google.cloud import bigquery, storage
import pandas as pd
from scipy.spatial import distance


PROJECT = "p9-reco-contenu"
DATASET = "p9_reco"

client = bigquery.Client(project=PROJECT)
client_gcs = storage.Client(project=PROJECT)
bucket = client_gcs.get_bucket('data-p9-reco')

articles_df = BytesIO(bucket.get_blob(
    'articles_medata_transformed.feather').download_as_bytes())
articles_df = pd.read_feather(articles_df).drop(['created_at_ts',
                                                 'publisher_id',
                                                 'words_count',
                                                 'category_id'],
                                                axis=1)


def get_liste_articles(user_id):
    """
    Récupère la liste des articles lu par l'utilisateur sur le dataset
    big query.
    """
    query = f"""
        SELECT article_id
        FROM `{PROJECT}.{DATASET}.train`
        WHERE user_id = {user_id}
        """
    query_job = client.query(query)
    result = query_job.result().to_dataframe()['article_id'].tolist()

    return result

def get_unread_articles(liste_articles):
    """
    Récupère la liste des articles non lu par le user, leur catégories
    et leur embedding.
    """
    ## Solution non-retenue car temps d'éxecution trop long et trop gourmand en
    ## sur Big Query
    # query = f"""
    #     SELECT article_id
    #     FROM `{PROJECT}.{DATASET}.train`
    #     WHERE user_id = {user_id}
    #     SELECT
    #         article_id,
    #         embeddings
    #     FROM
    #         `{PROJECT}.{DATASET}.article_metadata`
    #     WHERE
    #         article_id NOT IN {liste_articles}
    #     """
    selection = articles_df[~articles_df.article_id.isin(liste_articles)].copy()

    return selection




@functions_framework.http
def cb_get_articles_id(request):
    """
    Renvoie une sélection de N articles correspondant au user
    Args:
        user_id : Id de l'utilisateur
        method = 'last' : Méthode de calcul de référence des préférences du user
                        Doit être 'last' ou 'mean'
        nb_results = 5 : Nombre de résultats renvoyé
    """

    try:
        user_id = request.args.get('user_id')
        user_id = int(user_id)
    except ValueError as e:
        raise ValueError("Le user_id doit être spécifié") from e

    method = request.args.get('method')
    if not method:
        warnings.warn('Pas de méthode précisée, dernier article lu utilisé.')
        method = 'last'
    elif method not in ['last', 'mean']:
        warnings.warn('Méthode précisée invalide, dernier article lu utilisé.')
        method = 'last'

    nb_results = request.args.get('nb_results')
    if not nb_results:
        warnings.warn('Nombre de résultats par défaut renvoyé (5)')
        nb_results = 5
    nb_results = int(nb_results)


    liste_articles = get_liste_articles(user_id)
    selection = get_unread_articles(liste_articles)
    if method == 'last':
        query_encoding = articles_df[articles_df['article_id']
                                     == liste_articles[-1]]['embeddings'].item()
    else:
        query_encoding = articles_df[articles_df['user_id']
                                       == user_id]['embeddings'].mean()

    selection['similarity_score'] = selection['embeddings'].apply(
        lambda x: 1 - distance.cosine(x, query_encoding))

    res = selection.sort_values(by=['similarity_score'], ascending=False)[
        ['article_id', 'similarity_score']].head(nb_results).to_json()
    res = {res['article_id'][k]: v for k, v in res['similarity_score'].items()}

    return res
