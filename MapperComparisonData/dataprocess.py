import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from cereeberus import ReebGraph, MapperGraph, Interleave


# read the data from json file

# def read_mapper(file_path):
#     """
#     Reads a Mapper graph from a JSON file and constructs a NetworkX graph.
#     Also computes the average L2 norm for each node to use as a height attribute.
    
#     Parameters:
#     - file_path: str, path to the JSON file containing the Mapper graph data.
    
#     Returns:
#     - G: networkx.Graph, the constructed graph with height attributes.
#     - heights: dict, mapping of node IDs to their average L2 norm heights."""

#     with open(file_path, 'r') as f:
#         mapper_data = json.load(f)

#     nodes = mapper_data['nodes']
#     edges = mapper_data['links']
#     data_points = mapper_data['data_points']

#     G = nx.Graph()
#     heights = {}
#     # compute the the avg L2 norm for each node
#     for node in nodes:
#         point_ids = node['vertices']
#         l2_norms = [data_points[str(pid)] for pid in point_ids]
#         avg_l2_norm = np.mean(l2_norms) 
#         heights[node['id']] = avg_l2_norm
#         # add node to graph with avg L2 norm as height attribute

#         G.add_node(node['id'], height=avg_l2_norm)

#     for edge in edges:
#         G.add_edge(edge['source'], edge['target'])

#     return G, heights

# #if the heights aren't integers, we need to scale them to integers

# def scale_heights(heights, scale_factor= 10):
#     """
#     Scales the height values to integers to increase resolution.
#     Modifies the heights dictionary in place.

#     Parameters:
#     - heights: dict, mapping of node IDs to their height values.
#     - scale_factor: int, factor by which to scale the heights.

#     Returns:
#     - heights: dict, updated mapping of node IDs to their scaled integer heights.
#     """
    
#     for node_id in heights:
#         heights[node_id] = int(heights[node_id] * scale_factor)

#     return heights

def read_mapper(file_path):


    with open(file_path, 'r') as f:
        mapper_data = json.load(f)

    nodes = mapper_data['nodes']
    edges = mapper_data['links']
    data_points = mapper_data['data_points']

    G = nx.Graph()

    # compute avarage L2 norm for each cube

    cube_points = {}
    for node in nodes:
        cube_id = node['id'].split('_')[0]  # Extract cube ID from node ID
        if cube_id not in cube_points:
            cube_points[cube_id] = []
        cube_points[cube_id].extend(node['vertices']) 

        # compute the the avg L2 norm for each cube

    cube_heights = {}

    for cube_id, point_ids in cube_points.items():
        l2_norms = [data_points[str(pid)] for pid in point_ids]
        avg_l2_norm = np.mean(l2_norms) 
        cube_heights[cube_id] = avg_l2_norm

    

    # assign each node the height of its cube
    heights = {}

    for node in nodes:
        cube_id = node['id'].split('_')[0]
        heights[node['id']] = cube_heights[cube_id]
        G.add_node(node['id'], f=cube_heights[cube_id])

    # add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'])

    # scale heights to integers
    heights = scale_heights(heights, scale_factor=10)
    
    return G, heights

def scale_heights(heights, scale_factor= 10):
    for node_id in heights:
        heights[node_id] = int(heights[node_id] * scale_factor)

    return heights

