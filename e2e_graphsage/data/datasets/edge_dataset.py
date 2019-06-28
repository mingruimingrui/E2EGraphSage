"""
"""

import random
from collections import Iterable

import numpy as np
import torch.utils.data


class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, adjacency_list, src_node_ids=None):
        formatted_adjacency_list = []
        for neighbors in adjacency_list:
            neighbors = np.array(neighbors, dtype=np.int64)
            neighbors = neighbors[neighbors != -1]
            neighbors = np.ascontiguousarray(neighbors)
            formatted_adjacency_list.append(neighbors)
        formatted_adjacency_list = np.array(formatted_adjacency_list)
        self.adjacency_list = formatted_adjacency_list

        if src_node_ids is None:
            self.src_node_ids = np.arange(len(adjacency_list)).astype(np.int64)
        else:
            self.src_node_ids = np.array(src_node_ids, dtype=np.int64)

        assert self.src_node_ids.max() < len(adjacency_list)

    def __len__(self):
        return len(self.src_node_ids)

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            src_node_ids = self.src_node_ids[idx]
            all_neighbors = self.adjacency_list[src_node_ids]
            neigh_node_ids = [
                random.choice(neighbors)
                for neighbors in all_neighbors
            ]

            src_node_id = torch.LongTensor(src_node_ids)
            neigh_node_id = torch.LongTensor(neigh_node_ids)

        else:
            src_node_id = self.src_node_ids[idx]
            neighbors = self.adjacency_list[src_node_id]
            neigh_node_id = random.choice(neighbors)

            src_node_id = torch.LongTensor(src_node_id)[0]
            neigh_node_id = torch.LongTensor(neigh_node_id)[0]

        return src_node_id, neigh_node_id
