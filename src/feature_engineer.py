import pandas as pd
import numpy as np
import math
import networkx as nx
from tqdm import tqdm
import pickle
from config import *

class FeatureEngineer:
    def __init__(self, graph, pagerank, katz, hits, wcc, adj_dict, svd_u, svd_v):
        self.graph = graph
        self.pagerank = pagerank
        self.katz = katz
        self.hits = hits
        self.wcc = wcc
        self.adj_dict = adj_dict
        self.svd_u = svd_u
        self.svd_v = svd_v

    def jaccard_followers(self, a, b):
        try:
            pred_a = set(self.graph.predecessors(a))
            pred_b = set(self.graph.predecessors(b))
            if len(pred_a) == 0 or len(pred_b) == 0:
                return 0
            intersection = len(pred_a.intersection(pred_b))
            union = len(pred_a.union(pred_b))
            return intersection / union if union > 0 else 0
        except:
            return 0

    def jaccard_followees(self, a, b):
        try:
            succ_a = set(self.graph.successors(a))
            succ_b = set(self.graph.successors(b))
            if len(succ_a) == 0 or len(succ_b) == 0:
                return 0
            intersection = len(succ_a.intersection(succ_b))
            union = len(succ_a.union(succ_b))
            return intersection / union if union > 0 else 0
        except:
            return 0

    def cosine_followers(self, a, b):
        try:
            pred_a = set(self.graph.predecessors(a))
            pred_b = set(self.graph.predecessors(b))
            if len(pred_a) == 0 or len(pred_b) == 0:
                return 0
            intersection = len(pred_a.intersection(pred_b))
            denom = math.sqrt(len(pred_a) * len(pred_b))
            return intersection / denom if denom > 0 else 0
        except:
            return 0

    def cosine_followees(self, a, b):
        try:
            succ_a = set(self.graph.successors(a))
            succ_b = set(self.graph.successors(b))
            if len(succ_a) == 0 or len(succ_b) == 0:
                return 0
            intersection = len(succ_a.intersection(succ_b))
            denom = math.sqrt(len(succ_a) * len(succ_b))
            return intersection / denom if denom > 0 else 0
        except:
            return 0

    def adar_index(self, a, b):
        try:
            common_neighbors = set(self.graph.successors(a)).intersection(
                set(self.graph.successors(b)))
            if len(common_neighbors) == 0:
                return 0
            adar_sum = 0
            for neighbor in common_neighbors:
                degree = len(list(self.graph.predecessors(neighbor)))
                if degree > 1:
                    adar_sum += 1 / np.log10(degree)
            return adar_sum
        except:
            return 0

    def shortest_path_length(self, a, b):
        try:
            length = nx.shortest_path_length(self.graph, source=a, target=b)
            return length
        except:
            return -1

    def same_wcc(self, a, b):
        try:
            if self.graph.has_edge(b, a):
                return 1
            for component in self.wcc:
                if a in component and b in component:
                    return 1
            return 0
        except:
            return 0

    def follows_back(self, a, b):
        return 1 if self.graph.has_edge(b, a) else 0

    def get_svd_features(self, node, component='u'):
        try:
            if node in self.adj_dict:
                idx = self.adj_dict[node]
                if component == 'u':
                    return self.svd_u[idx].tolist()
                else:
                    return self.svd_v[idx].tolist()
            return [0] * 6
        except:
            return [0] * 6

    def engineer_features(self, df):
        print("Engineering features...")
        features = pd.DataFrame(index=df.index)

        # Basic node features
        print("Extracting basic node features...")
        features['num_followers_s'] = df['source_node'].apply(
            lambda x: len(list(self.graph.predecessors(x))))
        features['num_followers_d'] = df['destination_node'].apply(
            lambda x: len(list(self.graph.predecessors(x))))
        features['num_followees_s'] = df['source_node'].apply(
            lambda x: len(list(self.graph.successors(x))))
        features['num_followees_d'] = df['destination_node'].apply(
            lambda x: len(list(self.graph.successors(x))))

        # Common neighbors
        print("common neighbors")
        features['inter_followers'] = df.apply(
            lambda row: len(set(self.graph.predecessors(row['source_node'])).intersection(
                set(self.graph.predecessors(row['destination_node'])))), axis=1)
        features['inter_followees'] = df.apply(
            lambda row: len(set(self.graph.successors(row['source_node'])).intersection(
                set(self.graph.successors(row['destination_node'])))), axis=1)

        # Similarity features
        print("similarity features ")
        features['jaccard_followers'] = df.apply(
            lambda row: self.jaccard_followers(row['source_node'], row['destination_node']),
            axis=1)
        features['jaccard_followees'] = df.apply(
            lambda row: self.jaccard_followees(row['source_node'], row['destination_node']),
            axis=1)
        features['cosine_followers'] = df.apply(
            lambda row: self.cosine_followers(row['source_node'], row['destination_node']),
            axis=1)
        features['cosine_followees'] = df.apply(
            lambda row: self.cosine_followees(row['source_node'], row['destination_node']),
            axis=1)

        # Centrality features
        print("centrality feature")
        features['pagerank_s'] = df['source_node'].apply(
            lambda x: self.pagerank.get(x, 0))
        features['pagerank_d'] = df['destination_node'].apply(
            lambda x: self.pagerank.get(x, 0))
        features['katz_s'] = df['source_node'].apply(
            lambda x: self.katz.get(x, 0))
        features['katz_d'] = df['destination_node'].apply(
            lambda x: self.katz.get(x, 0))
        features['hits_hub_s'] = df['source_node'].apply(
            lambda x: self.hits['hubs'].get(x, 0))
        features['hits_hub_d'] = df['destination_node'].apply(
            lambda x: self.hits['hubs'].get(x, 0))
        features['hits_auth_s'] = df['source_node'].apply(
            lambda x: self.hits['authorities'].get(x, 0))
        features['hits_auth_d'] = df['destination_node'].apply(
            lambda x: self.hits['authorities'].get(x, 0))

        # Path and component features
        print("Extracting path and component features...")
        features['shortest_path'] = df.apply(
            lambda row: self.shortest_path_length(row['source_node'], row['destination_node']),
            axis=1)
        features['same_wcc'] = df.apply(
            lambda row: self.same_wcc(row['source_node'], row['destination_node']),
            axis=1)

        # Link prediction features
        print("Extracting link prediction features...")
        features['adar_index'] = df.apply(
            lambda row: self.adar_index(row['source_node'], row['destination_node']),
            axis=1)
        features['follows_back'] = df.apply(
            lambda row: self.follows_back(row['source_node'], row['destination_node']),
            axis=1)

        # SVD features
        print("Extracting SVD features...")
        svd_s = df['source_node'].apply(lambda x: self.get_svd_features(x, 'u'))
        svd_d = df['destination_node'].apply(lambda x: self.get_svd_features(x, 'u'))

        for i in range(6):
            features[f'svd_u_s_{i+1}'] = svd_s.apply(lambda x: x[i])
            features[f'svd_u_d_{i+1}'] = svd_d.apply(lambda x: x[i])

        return features
