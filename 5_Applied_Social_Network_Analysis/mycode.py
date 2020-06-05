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











































