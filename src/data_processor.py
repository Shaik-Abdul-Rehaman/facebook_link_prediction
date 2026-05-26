import pandas as pd
import numpy as np
import networkx as nx
import pickle
import csv
import random
import math
from pathlib import Path
from tqdm import tqdm
import os
from config import *

class DataProcessor:
    def __init__(self):
        self.graph = None
        self.train_graph = None
        self.missing_edges = None
        self.wcc = None
        self.pagerank = None
        self.katz = None
        self.hits = None
        self.svd_dict = {}
        self.svd_u = None
        self.svd_v = None
        self.adj_dict = {}

    def load_or_create_graph(self):
        if os.path.exists(GRAPH_FILE):
            print(f"Loading graph from {GRAPH_FILE}")
            self.graph = pickle.load(open(GRAPH_FILE, 'rb'))
        else:
            print("Creating graph from train.csv")
            traincsv = pd.read_csv(TRAIN_CSV)
            print(f"Train CSV shape: {traincsv.shape}")
            print(f"Duplicates: {sum(traincsv.duplicated())}")
            traincsv.to_csv(GRAPH_CSV, header=False, index=False)
            self.graph = nx.read_edgelist(str(GRAPH_CSV), delimiter=',',
                                          create_using=nx.DiGraph(), nodetype=int)
            pickle.dump(self.graph, open(GRAPH_FILE, 'wb'))

        self.train_graph = self.graph.copy()
        return self.graph

    def load_or_create_missing_edges(self, limit=9437519):
        if os.path.exists(MISSING_EDGES_FILE):
            print(f"Loading missing edges from {MISSING_EDGES_FILE}")
            self.missing_edges = pickle.load(open(MISSING_EDGES_FILE, 'rb'))
        else:
            print("Generating negative edges (missing edges)")
            r = csv.reader(open(GRAPH_CSV, 'r'))
            edges = {}
            for edge in r:
                edges[(edge[0], edge[1])] = 1

            self.missing_edges = set([])
            max_node = max(self.graph.nodes())
            pbar = tqdm(total=limit)

            while len(self.missing_edges) < limit:
                a = random.randint(1, max_node)
                b = random.randint(1, max_node)
                tmp = edges.get((a, b), -1)

                if tmp == -1 and a != b:
                    try:
                        if nx.shortest_path_length(self.graph, source=a, target=b) > 2:
                            if (a, b) not in self.missing_edges:
                                self.missing_edges.add((a, b))
                                pbar.update(1)
                    except:
                        if (a, b) not in self.missing_edges:
                            self.missing_edges.add((a, b))
                            pbar.update(1)
            pbar.close()
            pickle.dump(self.missing_edges, open(MISSING_EDGES_FILE, 'wb'))

        return self.missing_edges

    def load_or_compute_centrality_measures(self):
        # PageRank
        if os.path.exists(PAGERANK_FILE):
            self.pagerank = pickle.load(open(PAGERANK_FILE, 'rb'))
        else:
            print("Computing PageRank...")
            self.pagerank = nx.pagerank(self.train_graph, alpha=0.85)
            pickle.dump(self.pagerank, open(PAGERANK_FILE, 'wb'))

        # Katz Centrality
        if os.path.exists(KATZ_FILE):
            self.katz = pickle.load(open(KATZ_FILE, 'rb'))
        else:
            print("Computing Katz centrality...")
            self.katz = nx.katz_centrality(self.train_graph, alpha=0.005, beta=1)
            pickle.dump(self.katz, open(KATZ_FILE, 'wb'))

        # HITS
        if os.path.exists(HITS_FILE):
            self.hits = pickle.load(open(HITS_FILE, 'rb'))
        else:
            print("Computing HITS...")
            hubs, authorities = nx.hits(self.train_graph, max_iter=100, tol=1e-08, normalized=True)
            self.hits = {'hubs': hubs, 'authorities': authorities}
            pickle.dump(self.hits, open(HITS_FILE, 'wb'))

        # Weakly Connected Components
        if os.path.exists(WCC_FILE):
            self.wcc = pickle.load(open(WCC_FILE, 'rb'))
        else:
            print("Computing weakly connected components...")
            self.wcc = list(nx.weakly_connected_components(self.train_graph))
            pickle.dump(self.wcc, open(WCC_FILE, 'wb'))

    def compute_svd_features(self, k=6):
        if os.path.exists(SVD_COMPONENTS_FILE):
            data = pickle.load(open(SVD_COMPONENTS_FILE, 'rb'))
            self.adj_dict = data['adj_dict']
            self.svd_u = data['svd_u']
            self.svd_v = data['svd_v']
        else:
            print("Computing SVD components...")
            from scipy.sparse.linalg import svds

            nodes = sorted(self.train_graph.nodes())
            self.adj_dict = {val: idx for idx, val in enumerate(nodes)}
            adj = nx.adjacency_matrix(self.train_graph, nodelist=nodes).astype('float32')
            U, s, V = svds(adj, k=k)
            self.svd_u = U
            self.svd_v = V

            pickle.dump({
                'adj_dict': self.adj_dict,
                'svd_u': self.svd_u,
                'svd_v': self.svd_v
            }, open(SVD_COMPONENTS_FILE, 'wb'))

    def prepare_training_data(self):
        print("Preparing training data...")
        self.load_or_create_graph()
        self.load_or_create_missing_edges()
        self.load_or_compute_centrality_measures()
        self.compute_svd_features()

        # Create positive and negative samples
        df_pos = pd.read_csv(TRAIN_CSV)
        df_neg = pd.DataFrame(list(self.missing_edges),
                              columns=['source_node', 'destination_node'])

        # Train-test split
        from sklearn.model_selection import train_test_split

        X_train_pos, X_test_pos = train_test_split(df_pos, test_size=0.2, random_state=9)
        X_train_neg, X_test_neg = train_test_split(df_neg, test_size=0.2, random_state=9)

        X_train = pd.concat([X_train_pos.assign(indicator_link=1),
                             X_train_neg.assign(indicator_link=0)], ignore_index=True)
        X_test = pd.concat([X_test_pos.assign(indicator_link=1),
                            X_test_neg.assign(indicator_link=0)], ignore_index=True)

        return X_train, X_test
