# Mapper Graph Export - JSON File Description

## Overview
This JSON file contains the largest connected component of a Mapper graph generated from neural network layer activations. The Mapper algorithm creates a topological summary of high-dimensional data by clustering points and connecting overlapping clusters.

## File Structure

### Root Level
```json
{
  "nodes": [...],           // Array of node objects
  "links": [...],           // Array of edge objects  
  "data_points": {...},     // Object mapping point IDs to L2 norms
  "metadata": {...}         // Export information and statistics
}
```

### 1. Nodes Array
Each node represents a cluster of data points:
```json
{
  "id": "cube1_cluster0",     // Unique node identifier
  "vertices": [3080, 3117, 283],  // Array of point IDs in this cluster
  "comp_id": 0                // Connected component ID
}
```

### 2. Links Array  
Each link connects two nodes that share data points:
```json
{
  "source": "cube1_cluster0",  // Source node ID
  "target": "cube2_cluster1",  // Target node ID
  "jcd_sim": 0.85              // Jaccard similarity (shared points / total points)
}
```

### 3. Data Points Object
Maps each data point to its L2 norm value:
```json
{
  "3080": 17.907749583232484,  // point_id: L2_norm_value
  "3117": 18.234567890123456,
  "283": 16.123456789012345
}
```

### 4. Metadata Object
Contains export information:
```json
{
  "component_id": 0,           // ID of the exported component
  "component_size": 45,        // Number of nodes in this component
  "dataset": "gmb_data_cia",   // Source dataset name
  "layer": "12",               // Neural network layer number
  "total_points": 1234         // Total number of data points
}
```

## Usage
- **Graph Analysis**: Use `nodes` and `links` for topological analysis
- **Point Mapping**: Use `data_points` to map point IDs to their L2 norm values
- **Component Info**: Use `metadata` to understand the export context

## File Naming Convention
Files are named as: `mapper_graph_layer_{layer_number}.json`

## Technical Notes
- Only the largest connected component is exported (by node count)
- Point IDs correspond to indices in the original dataset
- L2 norms are computed from the original activation vectors
- Jaccard similarity measures overlap between connected nodes
