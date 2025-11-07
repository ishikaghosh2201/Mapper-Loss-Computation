import kmapper as km
from cereeberus import MapperGraph
from sklearn.cluster import DBSCAN
from sklearn import datasets
import networkx as nx
import numpy as np


def create_basic_mapper(sample):
    # Create a KeplerMapper object
    kM = km.KeplerMapper(verbose=False)

    # Y axis as lens
    lens = kM.project(sample, projection=[1], scaler=None)

    # create the cover
    cover = km.Cover(n_cubes=5, perc_overlap=0.2)

    km_graph = kM.map(lens,
                    sample,
                    cover=cover,
                    clusterer=DBSCAN(eps=0.1, min_samples=3))
    
    # convert to networkx graph
    nM = km.to_networkx(km_graph)

    for node_id, cluster_indices in km_graph['nodes'].items():
        
        # get ycoordinates of points in the cluster
        y_coords = [sample[i][1] for i in cluster_indices]

        # calculate the the mean height
        height = int(np.mean(y_coords)*20)

        # set the height as a node attribute
        nM.nodes[node_id]['fx'] = height

    return nM

def normalize_node_heights(G, resolution=10):

    # get the heights 
    heights = nx.get_node_attributes(G, 'fx')

    min_height = min(heights.values())
    max_height = max(heights.values())

    for node, height in heights.items():
        # normalize to [0, 1]
        norm_height = (height - min_height) / (max_height - min_height)

       # scale to resolution
        scaled_height = int(norm_height * resolution)

        # update the node attribute
        G.nodes[node]['fx'] = scaled_height

    return G

def generate_mapper(sample):

    G1 = create_basic_mapper(sample)
    G2 = normalize_node_heights(G1)

    M = MapperGraph(G2)

    return M

def plot_mapper(mapper, sample):

    #plot the mapper and the sample side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Mapper Graph")
    mapper.draw(ax=axes[0])
    axes[1].set_title("Sample Points")
    axes[1].scatter(sample[:, 0], sample[:, 1], s=1)
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 2)
    axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.show()

