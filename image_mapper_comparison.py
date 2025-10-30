### Workflow to compare mapper graphs generated from image data ###
##################################################################


### ---------------------------------###
### Import necessary libraries ###
### ---------------------------------###
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
from itertools import combinations_with_replacement
import multiprocessing as mp

from cereeberus import ReebGraph, Interleave, MapperGraph


def resize_image(image, target_height=None):
    """
    Resize the image to a target height while maintaining the aspect ratio.

    Parameters
        image : PIL image
        target_height : int

    Returns
        PIL image
    """
    # when target_height is None, return the original image
    if target_height is None:
        return image

    width, height = image.size
    aspect_ratio = width / height
    new_width = int(aspect_ratio * target_height)
    new_image = image.resize((new_width, target_height))
    return new_image


def rotate_image(image, angle=90):
    """
    Rotate the image by a given angle. The default angle is 90 degrees.

    Parameters
        image : PIL image

    Returns
        PIL image
    """
    return image.rotate(angle)


def save_rotated_image(image_path, angle=90):
    """
    Rotate the image and save it to an output path.

    Parameters
        image_path : str
        angle : int

    Returns
        None
    """
    image = Image.open(image_path)
    image = rotate_image(image, angle)
    # save to the same file path
    image.save(image_path)


### ---------------------------------###
### Load image data ###
### ---------------------------------###
def get_point_cloud_from_image(image_path, rotate_angle=None):
    """
    Get the point clouds from the image. This returns the coordinates of the white pixels in the image.
    
    Parameters
        image_path : str

    Returns
       numpy array
    """
   # Load the image and resize it
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize the image
    image = resize_image(image)

    # Rotate the image
    if rotate_angle is not None:
        image = rotate_image(image, rotate_angle)

    # Convert to numpy array
    image = np.array(image)

    # get what the maximum pixel values are
    max_pixel = np.max(image)

    # save where the white pixels are and save that point cloud
    points = np.column_stack(np.where(image == max_pixel))

    return points


def import_image_paths(image_dir):
    """
    Import the image paths from a directory. We remove the images where there are multiple images with almost same name, like apple-1.gif and apple-1 2.gif. We only keep the first one.

    Parameters
        image_dir : str

    Returns
        list
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith(
        '.gif') and " " not in f]  # Remove files with spaces
    image_paths = [os.path.join(image_dir, f)
                   for f in image_files]  # Convert to full paths

    # order the image paths by the image names
    image_paths = sorted(image_paths, key=lambda x: int(
        os.path.splitext(os.path.basename(x))[0].split('-')[-1]))
    return image_paths

### ---------------------------------###
### Create Mapper graph from image data ###
### ---------------------------------###


def mapper_of_image(points, lens="proj"):  # , num_intervals=5, overlap=0.3):
    """
    Create a Mapper graph from the point clouds of the image.

    Parameters
        points : numpy array
        func_val : numpy array
        num_intervals : int
        overlap : float

    Returns     
        networkx graph
    """
    # Create the mapper graph
    mapper = km.KeplerMapper(verbose=False)

    # set the lens
    if lens == "proj":
        lens = mapper.project(points, projection=[1], scaler=None)

    # # create lens
    # lens = mapper.project(points, projection=[1], scaler=None) # projection=[1] means we are using the y-coordinate as the lens

    # Create the graph
    graph = mapper.map(lens,
                       points,
                       clusterer=DBSCAN(eps=3), cover=km.Cover(n_cubes=5, perc_overlap=0.3))  # why are we using these parameters?
    # eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other
    # min_samples is the number of samples in a neighborhood for a point to be considered as a core point
    # n_cubes is the number of intervals in the lens
    # perc_overlap is the overlap between the intervals

    # convert to networkx graph
    G = km.adapter.to_nx(graph)

    vert_name_counter = 0  # counter for the vertex names. increases by 1 for each new vertex

    # Assign heights to each cluster based on y-coordinates of points
    for node_id, cluster_points_indices in graph['nodes'].items():
        # Get the y-coordinates of points in the cluster
        y_coords = [points[i][1] for i in cluster_points_indices]

        # Calculate height (e.g., mean y-coordinate)
        height = int(np.mean(y_coords)) if y_coords else 0

        # rename node_id to simplify the graph. Put v+height as the new node_id
        G = nx.relabel_nodes(G, {node_id: f"v{vert_name_counter}"})
        G.nodes(data=True)

        # Assign the height as a node attribute
        # Remove all existing attributes
        G.nodes[f"v{vert_name_counter}"].clear()
        G.nodes[f"v{vert_name_counter}"]['fx'] = height
        vert_name_counter += 1

    return G


def normalize_node_heights(G, min_target=0, max_target=1, precision=2):
    """
    Normalize the heights of the nodes in the graph. The heights are normalized to be between 0 and 1. This is done to make differnt mapper graphs comparable. The precision parameter is used to round the heights to a certain number of decimal places.

    Parameters
        G : networkx graph
        min_target : int
        max_target : int

    Returns     
        networkx graph
    """
    # Get the maximum height
    heights = nx.get_node_attributes(G, 'fx')

    min_height = min(heights.values())
    max_height = max(heights.values())

    # Normalize the heights
    for node, height in heights.items():
        # Normalize the height to be between the given target range. Can change the resoltion of the heights later.
        normalized_height = ((height - min_height) / (max_height -
                             min_height)) * (max_target - min_target) + min_target
        normalized_height = round(normalized_height, precision)
        G.nodes[node]['fx'] = normalized_height

    return G

### ---------------------------------###
### Convert networkx graph the mapper structure we can work with ###
### ---------------------------------###


def generate_mapper_from_graph(G, resolution=10):
    """
    Generate a MapperGraph object from a networkx graph.

    Parameters
        G : networkx graph
        resolution : int # resolution to make the heights integers

    Returns
        MapperGraph
    """
    # Create a MapperGraph object from kepler mapper output

    # set the node heights as a dictionary
    node_heights = nx.get_node_attributes(G, 'fx')

    # multiply by resolution to make the heights integers. Default resolution is 10
    for node in node_heights:
        node_heights[node] = int(node_heights[node]*resolution)

    mapperG = MapperGraph(G, node_heights)


    # # Convert to a ReebGraph
    # reebG = ReebGraph(G, verbose=False)

    # # make sure that the function values are integers in reeb graph
    # for node in reebG.nodes():
    #     # multiply by resolution to make the heights integers. Default resolution is 20
    #     reebG.f[node] = int(reebG.f[node]*resolution)
    # # Convert to a MapperGraph

    # mapperG = reebG.to_mapper()
    return mapperG

def generate_mapper(impath, resolution=10):
    """
    Generate a MapperGraph object from an image path.

    Parameters
        impath : str
        resolution : int # resolution to make the heights integers

    Returns
        MapperGraph
    """
    # Load the image and get the point clouds
    points = get_point_cloud_from_image(impath)

    # Create a mapper graph from the point clouds
    G = mapper_of_image(points)

    # Normalize the heights of the nodes
    G = normalize_node_heights(G)

    # Generate a MapperGraph object from the networkx graph
    mapperG = generate_mapper_from_graph(G, resolution)

    return mapperG

### ---------------------------------###
### Plotting functions ###
### ---------------------------------###


def plot_image_and_mapper(G, points, mapperG):
    """
    Plot the image and the mapper graph side by side.

    Parameters
        G : networkx graph
        points : numpy array
        mapperG : MapperGraph

        Returns
        None
        """

    # Plot both the point cloud and the mapper graph side by side

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the point cloud
    ax[0].scatter(points[:, 0], points[:, 1], s=1)
    ax[0].set_title("Point Cloud")
    ax[0].grid(True)  # Add a grid for better visualization

    # Plot the mapper graph
    pos = nx.spring_layout(G)
    for node_id in G.nodes:
        if 'fx' in G.nodes[node_id]:  # Ensure 'fx' exists
            pos[node_id][1] = G.nodes[node_id]['fx']

    nx.draw(G, pos=pos, node_size=50, with_labels=True, font_size=12,
            node_color='lightblue', ax=ax[1])
    ax[1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax[1].grid(True)  # Add a grid for better visualization
    ax[1].set_title("Mapper Graph")

    # save the plot
    plt.savefig("image_mapper_comparison.png")
    plt.show()
    # # Plot the mapper in our format
    # mapperG.draw(with_labels=True, ax=ax[2])
    # ax[2].set_title("Mapper Graph with integer subdivisions")
    # plt.axis('on')
    # plt.show()


### ---------------------------------###
### Create the image processing pipeline ###
### ---------------------------------###


def pairwise_image_to_mapper_comparison(image_path1, image_path2, resolution=10, plot=False, verbose=False, lens="proj"):
    # Load the image and get the point clouds
    first_points = get_point_cloud_from_image(image_path1)
    second_points = get_point_cloud_from_image(image_path2)

    # Create a mapper graph from the point clouds
    G1 = mapper_of_image(first_points, lens=lens)
    G2 = mapper_of_image(second_points, lens=lens)

    # Normalize the heights of the nodes
    G1 = normalize_node_heihgts(G1)
    G2 = normalize_node_heihgts(G2)

    # Generate a MapperGraph object from the networkx graph
    mapperG1 = generate_mapper(G1, resolution)
    mapperG2 = generate_mapper(G2, resolution)

    if plot:
        plot_image_and_mapper(G1, first_points, mapperG1)
        plot_image_and_mapper(G2, second_points, mapperG2)

    # Find the optimal n which minimizes the n+loss value
    bb = resolution
    result = find_optimal_upper_bound_es(
        mapperG1, mapperG2, bb, verbose=verbose)

    return result


def extract_image_names(image_paths):
    """
    Extract the image names from the image paths.

    Parameters
        image_paths : list

    Returns
        tuple
    """
    image_names = tuple(os.path.basename(image_path)
                        for image_path in image_paths)

    return image_names

# define a worker function to run the pairwise_image_to_mapper_comparison function in parallel


def worker(pair):
    image_1 = pair[0]
    image_2 = pair[1]
    return pairwise_image_to_mapper_comparison(image_1, image_2)


def parallel_mapper_comparison(image_paths, resolution=10, plot=False, verbose=False, num_procs=None, result_key='best_upper_bound'):
    """
    Run the pairwise_image_to_mapper_comparison function in parallel for multiple image pairs.

    Parameters
        image_paths : list
        resolution : int
        plot : bool
        verbose : bool
        num_procs : int

    Returns
        dict
    """
   # generate pariwise combinations of the image paths
    image_pairs = list(combinations_with_replacement(image_paths, 2))

    # Run the worker function in parallel
    if num_procs is None:
        num_procs = mp.cpu_count()  # use all available cores

    if verbose:
        print(f"Running with {num_procs} processes")

    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(worker, image_pairs)

    # get image pair names
    image_pair_names = list(map(extract_image_names, image_pairs))

    # convert the results to a dictionary
    all_results_dict = dict(zip(image_pair_names, results))

    # output only the result corresponding to the result_key
    results_dict = {image_pair: result[result_key]
                    for image_pair, result in all_results_dict.items()}

    return results_dict
