# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:44:56 2020

@author: sony
"""
import networkx as nx

G = nx.Graph()
G
G.add_edge('A','B')
G.add_edge('B','C')


G = nx.Graph()
G
G.add_edge('A','B', relation = 'family' , weight = 6)
G.add_edge('B','C' , relation= 'friend' , weight = 13)
G.add_edge('D','C' , relation= 'enemy' , weight = 6)
G.edges()
G.edges(data = True)
G.edges(data = 'weight')
G.edges['A','B']
G.get_edge_data('A','B')
G.get_edge_data('A','C')
G.edges['A','B']['relation']


G = nx.Graph()
G.add_edge('B','C' , relation= 'friend' , h = 13)
G.add_edge('B','C' , relation= 'friend' , hr = 13)
G.add_edge('B','C' , relation= 'enemy' , hrqy= 13)
G.edges(data= True)



#Directed Graph

G = nx.DiGraph()
G.add_edge('A','B', relation = 'family' , weight = 6)
G.add_edge('B','C' , relation= 'friend' , weight = 13)
G.add_edge('D','C' , relation= 'enemy' , weight = 6)

G.edges()
G.edges(data=True)


#MultiGraph 
G = nx.MultiGraph()
G.add_edge('A','B', relation = 'family' , weight = 6)
G.add_edge('B','A' , relation= 'friend' , weight = 13)
G.add_edge('B','C' , relation= 'enemy' , weight = 6)

G.edges()
G.edges(data=True)
G.edges['A','B',1]
G.edges['A']
G.get_edge_data('A','B')
G.get_edge_data('A','B')['weight']
G.get_edge_data('A','B')[0]['weight']

# Multi Directed Graph
G = nx.MultiDiGraph()
G.add_edge('A','B' , relation = 'family' , weight = 6)
G.add_edge('A','B' , relation= 'friend' , weight = 13)
G.add_edge('B','C' , relation= 'enemy' , weight = 6)

G.nodes()
G.add_node('A', role = 'trader')
G.add_node('B', role = 'manager')
G.add_node('C', role = 'trader')
G.nodes()
G.nodes(data = True)
G.node['A']


'''
Bipartite Graph
'''
from networkx.algorithms import bipartite

B = nx.Graph()
B.add_nodes_from(['A','B','C','D','E'], bipartite = 0 )
B.add_nodes_from([1,2,3,4], bipartite = 1 )
B.add_edges_from([('A',1),('B',1),('C',1),('C',3),('D',2)])
bipartite.is_bipartite(B)
nx.draw_networkx(B)


B.add_edge('A','B')

bipartite.is_bipartite(B)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import bipartite
B = nx.Graph()
B.add_edges_from([('A', 'G'),
                  ('A','I'),
                  ('B','H'),
                  ('C', 'G'),
                  ('C', 'I'),
                  ('D', 'H'),
                  ('E', 'I'),
                  ('F', 'G'),
                  ('F', 'J')])

X1 = set(['A', 'B', 'C', 'D', 'E', 'F'])
P = bipartite.weighted_projected_graph(B, X1)
P.get_edge_data('A','C')['weight']

nx.draw_networkx(B)
bipartite.is_bipartite(B)

'''
WEEK2
'''

import networkx as nx

G = nx.Graph()
G.add_edges_from([('A','K'),('A','B'), ('A','C'),('B','C') ,('B','K'),('C','E'),('C','F'),('D','E'),
                  ('E','F'),('E','H'),('F','G'),('I','J')])

nx.clustering(G,'A')
nx.clustering(G,'J')
nx.average_clustering(G)

G = nx.Graph()
G.add_edges_from([('A','K'),('A','B'),('B','C') ,('B','K'),('C','E'),('C','F'),('D','E'),
                  ('E','F'),('E','H'),('F','G'),('I','J'),('E','I')])

nx.shortest_path(G,'A','H')
nx.shortest_path_length(G,'A','H')
nx.shortest_path_length(G,'E','J')

T = nx.bfs_tree(G,'A')
T.edges()

nx.shortest_path_length(G,'A')
nx.eccentricity(G)
nx.radius(G)
nx.periphery(G)
nx.center(G)

nx.is_connected(G)
nx.number_connected_components(G)
nx.connected_components(G)
sorted(nx.connected_components(G))
nx.node_connected_component(G,'A')

G = nx.DiGraph()
G.add_edges_from([('A','K'),('A','B'),('B','C') ,('B','K'),('C','E'),('C','F'),('D','E'),
                  ('E','F'),('E','H'),('F','G'),('I','J'),('E','I')])
nx.is_strongly_connected(G)
nx.is_weakly_connected(G)
sorted(nx.strongly_connected_components(G))
sorted(nx.weakly_connected_components(G))

#Network Robustness
G = nx.Graph()
G.add_edges_from([('A','K'),('A','B'),('B','C') ,('B','K'),('C','E'),('C','F'),('D','E'),
                  ('E','F'),('E','H'),('F','G'),('I','J'),('E','I')])
nx.node_connectivity(G)
nx.minimum_node_cut(G)
nx.edge_connectivity(G)
nx.minimum_edge_cut(G)

nx.node_connectivity(G,'A','D')
nx.minimum_node_cut(G,'A','D')
nx.edge_connectivity(G,'A','D')
nx.minimum_edge_cut(G,'A','D')

'''
WEEK3
'''
import pandas as pd

#Degree centrality
G = nx.karate_club_graph()
G.nodes()
G = nx.convert_node_labels_to_integers(G, first_label = 1)
nx.degree_centrality(G)
pd.Series(nx.degree_centrality(G)).nlargest(n=5)

#closeness centrality
nx.closeness_centrality(G)
pd.Series(nx.closeness_centrality(G)).nlargest(n=5)

(len(G.nodes())-1)/sum(nx.shortest_path_length(G,32).values())

sorted(nx.shortest_path_length(G,32).items(), key = lambda x : x[0])

G = nx.Graph()
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,7)])
nx.betweenness_centrality(G,normalized =False)

G = nx.Graph()
G.add_edges_from([(1,2),(2,3),(3,4),(2,4)])
nx.betweenness_centrality(G,normalized = False)


G = nx.karate_club_graph()
G.nodes()
G = nx.convert_node_labels_to_integers(G, first_label = 1)
nx.betweenness_centrality(G)
pd.Series(nx.betweenness_centrality(G, normalized = False)).nlargest(n=5)
pd.Series(nx.betweenness_centrality(G)).nlargest(n=5)

import operator
sorted(nx.betweenness_centrality(G, normalized = False).items(), key = operator.itemgetter(1), reverse = True)[:5]

pd.Series(nx.betweenness_centrality(G, normalized = False, k =10)).nlargest(n=5)

pd.Series(nx.betweenness_centrality_subset(G,[34,33,30,21,16,27,15,23,10],[1,4,13,11,6,12,17,7],normalized = True)).nlargest(n=5)

pd.Series(nx.edge_betweenness_centrality(G)).sort_values(ascending = False)[:5]
pd.Series(nx.edge_betweenness_centrality(G)).nlargest()
nx.edge_betweenness_centrality(G)

pd.Series(nx.edge_betweenness_centrality_subset(G,[34,33,30,21,16,27,15,23,10],[1,4,13,11,6,12,17,7],normalized = True)).nlargest(n=5)

# Page Rank

pd.Series(nx.pagerank(G,)).nlargest(n=5)

G = nx.DiGraph()
g1 = [(word[0],word[1]) for word in 'AB BC CB BD DC DA DE EA'.split()]
G.add_edges_from(g1)
nx.pagerank(G)

G = nx.DiGraph()
g2 = [(word[0],word[1]) for word in 'AB BC CB BD DC DA DE EA BF FG GF BG'.split()]
G.add_edges_from(g2)
nx.pagerank(G, alpha = 0.8)

pd.Series(nx.hits(G)[0]).nlargest(n=5)
pd.Series(nx.hits(G)[1]).nlargest(n=5)


'''
QUIZ
'''

G = nx.Graph()
G.add_edges_from([('A','B'),('B','D'),('A','C'),('C','D'),('C','E'),('D','E'),('D','G'),('E','G'),('G','F')])
nx.degree_centrality(G)['D']
nx.closeness_centrality(G)['G']
nx.betweenness_centrality(G)['G']
nx.betweenness_centrality(G, normalized = False)['G']


(5)/(3*5)
nx.edge_betweenness_centrality(G, normalized =False,)

G = nx.DiGraph()
G.add_edges_from([('B','A'),('A','B'),('A','C'),('C','D'),('D','C')])
nx.pagerank(G,alpha = .95,max_iter = 100)['D']












