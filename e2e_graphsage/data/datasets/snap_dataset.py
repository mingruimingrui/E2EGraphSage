
import snap
import random
import numpy as np
import torch.utils.data
from collections import Iterable


class SnapDataset(torch.utils.data.Dataset):
    def __init__(self, G, src_nodeids=None):
        valid_nodeids = set()
        for node in G.Nodes():
            if node.GetDeg() > 0:
                valid_nodeids.add(node.GetId())

        if src_nodeids is None:
            src_nodeids = list(valid_nodeids)
            src_nodeids.sort()

        for nodeid in src_nodeids:
            assert nodeid in valid_nodeids

        self.G = G
        self.directed = isinstance(G, snap.PNGraph)
        self.src_nodeids = np.array(src_nodeids)

    def __len__(self):
        return len(self.src_nodeids)

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            left_nodeids = self.src_nodeids[idx]
            right_nodeids = []

            if self.directed:
                for left_nodeid in left_nodeids:
                    left_node = self.G.GetNI(left_nodeid)
                    num_out_nodes = left_node.GetOutDeg()
                    right_nodeid = left_node.GetOutNId(
                        random.randint(0, num_out_nodes - 1))
                    right_nodeids.append(right_nodeid)
            else:
                for left_nodeid in left_nodeids:
                    left_node = self.G.GetNI(left_nodeid)
                    num_out_nodes = left_node.GetDeg()
                    right_nodeid = left_node.GetNbrNId(
                        random.randint(0, num_out_nodes - 1))
                    right_nodeids.append(right_nodeid)

            left_out = left_nodeids
            right_out = np.array(right_nodeids)
            # left_out = left_nodeids.tolist()
            # right_out = right_nodeids
        else:
            left_nodeid = self.src_nodeids[idx]
            left_node = self.G.GetNI(left_nodeid)

            if self.directed:
                num_out_nodes = left_node.GetOutDeg()
                right_nodeid = left_node.GetOutNId(
                    random.randint(0, num_out_nodes - 1))
            else:
                num_out_nodes = left_node.GetDeg()
                right_nodeid = left_node.GetNbrNId(
                    random.randint(0, num_out_nodes - 1))

            left_out = left_nodeid
            right_out = right_nodeid
        return left_out, right_out
