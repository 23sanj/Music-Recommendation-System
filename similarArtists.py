import pandas as pd
import numpy as np
import requests
import math
import os
import json
import objectpath
from PIL import Image
import shutil


import string
from fuzzywuzzy import fuzz

import requests
import xml.etree.ElementTree as ET

from sklearn.preprocessing import normalize
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

from math import sqrt
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import matplotlib


from flask import jsonify, Blueprint
from flask_restful import Api, Resource, reqparse

user_file='/Users/i856168/PycharmProjects/Music_Recommendation_Project/data/10000.txt'
song_file='/Users/i856168/PycharmProjects/Music_Recommendation_Project/data/song_data.csv'
artist_file ='/Users/i856168/PycharmProjects/Music_Recommendation_Project/data/artist_data.txt'


def read_data(user_file,song_file,artist_file):
    #Reading users one by one
    users_df = pd.read_table(user_file,header=None)
    users_df.columns = ['user_id', 'song_id', 'listen_count']
    #Reading songs one by one
    songs_df = pd.read_csv(song_file)

    #Reading artists one by one
    artist_df = pd.read_table(artist_file,delimiter = '<SEP>',header=None)
    artist_df.columns = ['artist_id', 'artist_mbid', 'track_id','artist_name']

    return users_df,songs_df,artist_df


def preprocess_data(users_df, songs_df, artist_df):
    # Merging the dataframes
    user_songs_df = pd.merge(users_df, songs_df.drop_duplicates(['song_id']), on="song_id", how="left")
    user_songs_df = pd.merge(user_songs_df, artist_df.drop_duplicates(['artist_name']), on="artist_name", how="left")

    del user_songs_df['track_id']

    # We want to get the total listen count for each user and artist combination.
    user_songs_df['total_listen_count'] = user_songs_df.groupby(['artist_name', 'user_id'])['listen_count'].transform(
        'sum')

    # drop duplicate pairs of uid-artists
    user_songs_df = user_songs_df.drop_duplicates(subset=['user_id', 'artist_name'])
    # Total listen count for every artist:
    user_songs_df['total_artist_plays'] = user_songs_df.groupby(['artist_name'])[['total_listen_count']].transform(
        'sum')

    return user_songs_df


def get_ratings_matrix(user_songs_df):
    new_df = user_songs_df[['user_id', 'artist_name', 'total_listen_count']]
    # Generating user id for user name and artist names
    nUsers = new_df.user_id.unique().shape[0]
    nItems = new_df.artist_name.unique().shape[0]
    print 'Number of users = ' + str(nUsers) + ' | Number of artists = ' + str(nItems)

    users = pd.Series(new_df.user_id.unique())
    ulabels, ulevels = pd.factorize(users)
    userDict = dict(zip(ulevels, ulabels))

    artists = pd.Series(new_df.artist_name.unique())
    alabels, alevels = pd.factorize(artists)
    artistDict = dict(zip(alevels, alabels))

    new_df['user'] = new_df['user_id'].map(userDict)
    new_df['artist'] = new_df['artist_name'].map(artistDict)

    new_df = new_df.pivot(index='artist_name', columns='user', values='total_listen_count').fillna(0)
    # Data-pruning: Removing ratings less 2
    new_df[new_df < 2] = 0

    ratings_matrix = csr_matrix(new_df.values)  # scipy sparse matrix

    return new_df, ratings_matrix

def fit_model(ratings_matrix):
    #Fitting the model: measures similarity bectween artist vectors by using cosine similarity.
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(ratings_matrix)#Fitting our sparse matrix to the knn model
    return model_knn

#Main program
def recommend(query_artist, k=10):
    users_df, songs_df, artist_df= read_data(user_file, song_file, artist_file)
    user_songs_df=preprocess_data(users_df, songs_df, artist_df)
    new_df, ratings_matrix= get_ratings_matrix(user_songs_df)
    knn_model =fit_model(ratings_matrix)

    query_index = None
    ratio_tuples = []
    res_artist_names = []

    for i in new_df.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 70:
            current_query_index = new_df.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))

    print 'Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples])

    try:
        query_index = max(ratio_tuples, key=lambda x: x[1])[2]  # get the index of the best artist match in the data
    except:
        print 'Your artist didn\'t match any artists in the data. Try again'
        return None

    distances, indices = knn_model.kneighbors(new_df.iloc[query_index, :].reshape(1, -1), n_neighbors=k + 1)

    for i in range(0, len(distances.flatten())):
        res_artist_name = new_df.index[indices.flatten()[i]]
        res_artist_dist = distances.flatten()[i]
        res_artist_names.append(res_artist_name)
        if i == 0:
            print 'Artists similar to {0}:\n'.format(new_df.index[query_index])
        else:
            print '{0}: {1}, with distance of {2}:'.format(i, res_artist_name, res_artist_dist)
    res_artist_names = [x for x in res_artist_names if not "/" in x]
    res_artist_names = [x for x in res_artist_names if x != query_artist]
    return res_artist_names,user_songs_df

def get_mbid_artist(artist_name,user_songs_df):
    mbid_artist=user_songs_df.loc[user_songs_df['artist_name']==artist_name]['artist_mbid'].values[0]
    return str(mbid_artist)

#Select the mbid for the simialr artists
def get_mbid_artists(res_artist_names,user_songs_df):
    query_mbids=[]
    artist_names=[]
    for x in res_artist_names:
        mbid_artist=get_mbid_artist(x,user_songs_df)
        if mbid_artist == 'nan':
            continue
        query_mbids.append(mbid_artist)
        artist_names.append(x)
        #get_poster_link(mbid_artist,poster_links)
    return artist_names,query_mbids


# Get releases mbids for each query
def get_mbid_release(artist_mbid):
    release_mbids = []
    url = 'https://musicbrainz.org/ws/2/release?artist=' + artist_mbid + '&inc=release-groups&limit=300'
    response = requests.get(url)
    with open('responseRel_xml.xml', 'wb') as f:
        f.write(response.content)
        # create element tree object
    tree = ET.parse('responseRel_xml.xml')
    # get root element
    root = tree.getroot()

    for release in root.iter('{http://musicbrainz.org/ns/mmd-2.0#}release'):
        release_mbids.append(release.get('id'))
    f.close()
    os.remove(f.name)
    return release_mbids

#Search for cover art
def get_cover_art_links(release_mbids,cover_art_links):
    c=1
    for release_mbid in release_mbids:
        if c==2: #Getting only 1 poster
            break
        url='http://coverartarchive.org/release/'+ release_mbid
        response = requests.get(url)
        r = requests.head(url)
        if r.status_code == 404:
            continue
        data = json.loads(response.content)
        tree_obj = objectpath.Tree(data)#ObjectPath is a library that provides ability to query JSON and nested structures of dicts and lists.
        cover_art_links.append(list(tree_obj.execute('$..image')))
        c =c+1
    return cover_art_links

def get_cover_art_artist(artist_mbid):
    release_mbids =get_mbid_release(artist_mbid)
    cover_art_links=[]
    get_cover_art_links(release_mbids,cover_art_links)
    return cover_art_links


def get_poster(cover_art_links,artist_name):
    url = cover_art_links[0][0]
    response = requests.get(url, stream=True)
    path = 'cover_art/'
    with open(path+artist_name +'.png', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return

def main(query_artist,k=10):
    res_artist_names,user_songs_df=recommend(query_artist,k=10)
    artists_name, artists_mbid = get_mbid_artists(res_artist_names, user_songs_df)
    i = 0
    for artist_mbid in artists_mbid:
        i = i + 1
        cover_art_links = get_cover_art_artist(artist_mbid)
        print (cover_art_links)
        if cover_art_links == []:
            continue
        Image.open(cover_art_links[0][0]).show()

    return res_artist_names


