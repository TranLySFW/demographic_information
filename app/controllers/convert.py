from main import app

import sqlite3
import os
import pathlib
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


@app.route('/api/convert-csv', methods=['GET', 'POST'])
def convert_db_to_csv():
    database_path, db_files_names, csv_files_names = count_db_files()

    for db_name, csv_name in zip(db_files_names, csv_files_names):
        empty_df, df, splited_vector = split_vector(database_path, db_name)
        if not empty_df:
            df = clustering_vector(df, splited_vector)
            save_as_csv(df, database_path, csv_name)
    return "Converted"


def count_db_files():
    """
    Get names of database files in database folder
    :return:
    """
    database_path = pathlib.Path(os.path.join(app.root_path, "database"))
    db_files_path = list(database_path.glob("*.db"))
    db_files_names = [str(elem.name) for elem in db_files_path]
    csv_files_names = [elem.split(".")[0] + ".csv" for elem in db_files_names]
    return str(database_path.absolute()), db_files_names, csv_files_names


def split_vector(database_path, db_name):
    """
    Get dataframe from db file
    :param database_path:
    :param db_name:
    :return:
    """
    conn = sqlite3.connect(os.path.join(database_path, db_name))
    df = pd.read_sql_query("SELECT * FROM predictor", conn)
    conn.close()
    splited_vector = ""
    if not df.empty:
        character_vector = df['vector'].map(lambda row: row[1: -2])
        splited_vector = character_vector.str.split(",", n=-1, expand=True)
    return df.empty, df, splited_vector


def renumber_id(iterable):
    """
    renumber output of Agglomerative
    :param iterable:
    :return:
    """
    seen = {}
    counter = 1
    for x in iterable:
        i = seen.get(x)
        if i is None:
            seen[x] = counter
            yield counter
            counter += 1
        else:
            yield i


def clustering_vector(df, df_splitted_vector, distance_threshold=0.08):
    """
    Cluster vectors into IDs
    :param df_splitted_vector:
    :return:
    """
    clustering = AgglomerativeClustering(n_clusters=None,
                                         affinity='cosine',
                                         linkage='average',
                                         distance_threshold=distance_threshold)
    clustering.fit(df_splitted_vector)
    total_clusters = clustering.n_clusters_
    id_clusters = np.array(list(renumber_id(clustering.labels_)))
    df.insert(3, 'id_clusters', id_clusters)
    return df


def save_as_csv(df, database_path, csv_name):
    """
    save dataframe as csv file
    :param df:
    :param csv_name:
    :return:
    """
    return df.to_csv(os.path.join(database_path, csv_name), index=False)



