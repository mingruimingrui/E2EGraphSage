"""
Module to perform BFS using an adjacency list
This method is based off the implementation from the original graphsage
and offers an naive but memory and speed efficient method of storing
a semi-large graph onto GPU memory
"""

import torch
import numpy as np
from six import integer_types
from collections import Iterable


class ToHierarchicalList(torch.nn.Module):
    def __init__(self, adjacency_list, input_features, expansion_rates):
        """
        Args:
            adjacency_list: More accurately a sampled adjacency list
                should be a (num_nodes, num_neighbors) shaped 2D array
                that represents a graph. To achieve an adjacency list of this
                uniform shape, some up/down sampling is required.
                As for the value of num_neighbors to choose, it is recommended
                for this value to be as large as the largest value of
                expansion_rates.
            input_features: The input feature for each node
                should be a (num_nodes, feature_size) shaped 2D array
                where each row in input_features should correspond to the
                same row in adjacency_list.
            expansion_rates: A list of sampling rates at each step
                eg. if given expansion_rates = [2, 3], then
                (1 * 2) 1-step neighbors will be sampled and
                (2 * 3) 2-step neighbors will be sampled.
        """
        super(ToHierarchicalList, self).__init__()

        assert isinstance(adjacency_list, torch.Tensor) or \
            isinstance(adjacency_list, np.ndarray), \
            'adjacency_list should be an array'
        adjacency_list = torch.from_numpy(adjacency_list)
        assert not adjacency_list.dtype.is_floating_point, \
            'adjacency_list should be an int array'

        assert isinstance(input_features, torch.Tensor) or \
            isinstance(input_features, np.ndarray), \
            'input_features should be an array'
        assert len(adjacency_list) == len(input_features), \
            'adjacency_list and input_features should have same length'

        assert isinstance(expansion_rates, Iterable), \
            'expansion_rates should be a list of positive integers'
        for r in expansion_rates:
            assert isinstance(r, integer_types) and r > 0, \
                'expansion_rates should be a list of positive integers'
        assert adjacency_list.size(1) >= max(expansion_rates), \
            'adjacency_list is too small for given expansion_rates'

        self.adjacency_list_size = adjacency_list.size(1)
        self.expansion_rates = expansion_rates

        self.adjacency_list = torch.nn.Parameter(
            adjacency_list.long(), requires_grad=False)
        self.input_features = torch.nn.Parameter(
            torch.from_numpy(input_features), requires_grad=False)

    def forward(self, src_nodeids):
        hierarchical_nodeids = [src_nodeids]
        hierarchical_input_features = [self.input_features[src_nodeids]]
        for r in self.expansion_rates:
            neighbor_nodeids = self.adjacency_list[hierarchical_nodeids[-1]]
            sampled_cols = torch.randperm(self.adjacency_list_size)[:r]
            neighbor_nodeids = neighbor_nodeids[:, sampled_cols]
            neighbor_nodeids = neighbor_nodeids.reshape(-1)
            hierarchical_nodeids.append(neighbor_nodeids)
            hierarchical_input_features.append(
                self.input_features[neighbor_nodeids])
        return hierarchical_nodeids, hierarchical_input_features
