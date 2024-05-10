import pandas as pd
import json

# Function to load the graph from a CSV file using Pandas, skipping the first row and column
def load_graph_from_csv(file_path):
    df = pd.read_csv(file_path, index_col=0)
    graph = df.values.tolist()
    node_names = df.index.tolist()  # List of node names
    return graph, node_names

# Recursive function to find all paths
def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    paths = []
    for i, connection in enumerate(graph[start]):
        if connection == 1 and i not in path:
            newpaths = find_all_paths(graph, i, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

# Function to get all paths and map node indices to names
def get_all_paths_with_names(graph, node_names):
    all_paths = []
    for start in range(len(graph)):
        for end in range(len(graph)):
            if start != end:
                paths = find_all_paths(graph, start, end)
                # Convert node indices to names
                named_paths = [[node_names[node] for node in path] for path in paths]
                all_paths.extend(named_paths)
    return all_paths

# Loading the graph from the CSV file
graph, node_names = load_graph_from_csv('graph.csv')

# Get all possible path sets with node names
all_path_sets = get_all_paths_with_names(graph, node_names)

# Save the paths to a JSON file
with open('path_sets.json', 'w') as json_file:
    json.dump(all_path_sets, json_file, indent=4)

# Now 'all_path_sets' are saved in 'path_sets.json'.
print("All paths have been saved to 'path_sets.json'.")
