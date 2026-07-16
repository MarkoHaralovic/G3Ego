import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
      sys.path.insert(0, project_root)

import networkx as nx
from graph_construction.graphs.graph_to_nx import full_action_graph_to_nx, cw_eccentricity
from graph_training.dataset.GraphDataset import (
   GraphDataset,
   return_train_val_samples,
)

graph_type = "full"
train_samples, val_samples, activity_to_idx = return_train_val_samples(pooling="concat")
data_path = "/path/to/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
train_dataset = GraphDataset(data_path, train_samples, activity_to_idx, graph_type)

max_shortest_paths = 0.0
global_effs = 0.0
i = 0
for idx in range(len(train_dataset)):
      full_action_graphs = train_dataset.__getitem__(idx)
      full_action_graphs = full_action_graphs['full_action_graphs']

      for full_action_graph in full_action_graphs.values():
            i += 1
            Gnx, cw_node_id = full_action_graph_to_nx(full_action_graph, directed=True)

            max_shortest_path = cw_eccentricity(Gnx, cw_node_id)
            G_undir = Gnx.to_undirected()
            if G_undir.number_of_nodes() < 2:
                  global_eff = 0.0
            else:
                  global_eff = nx.global_efficiency(G_undir)
            max_shortest_paths += max_shortest_path
            global_effs += global_eff
        
max_shortest_paths /= i
global_effs /= i
   
graph_type = "pruned"
train_samples, val_samples, activity_to_idx = return_train_val_samples(pooling="concat")
data_path = "/path/to/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
train_dataset = GraphDataset(data_path, train_samples, activity_to_idx, graph_type)

max_shortest_paths_pruned = 0.0
global_effs_pruned = 0.0
i = 0
for idx in range(len(train_dataset)):
      full_action_graphs = train_dataset.__getitem__(idx)
      full_action_graphs = full_action_graphs['full_action_graphs']

      for full_action_graph in full_action_graphs.values():
            Gnx, cw_node_id = full_action_graph_to_nx(full_action_graph, directed=True)
            i += 1
            max_shortest_path = cw_eccentricity(Gnx, cw_node_id)
            G_undir = Gnx.to_undirected()
            if G_undir.number_of_nodes() < 2:
                  global_eff = 0.0
            else:
                  global_eff = nx.global_efficiency(G_undir)
            max_shortest_paths_pruned += max_shortest_path
            global_effs_pruned += global_eff

max_shortest_paths_pruned /= i
global_effs_pruned /= i

print("Full graphs : ")
print(f"Average max_shortest_paths : {max_shortest_paths}, avg global_effs : {global_effs}")
print()
print("Pruned graphs : ")
print(f"Average max_shortest_paths : {max_shortest_paths_pruned}, avg global_effs : {global_effs_pruned}")
