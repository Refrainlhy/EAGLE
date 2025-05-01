import torch
import random
import numpy as np
import pandas as pd

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.tbatch = None
        self.n_batch = 0

    def sample(self, ratio):
        data_size = self.n_interactions
        sample_size = int(ratio * data_size)
        sample_inds = random.sample(range(data_size), sample_size)
        sample_inds = np.sort(sample_inds)
        sources = self.sources[sample_inds]
        destination = self.destinations[sample_inds]
        timestamps = self.timestamps[sample_inds]
        edge_idxs = self.edge_idxs[sample_inds]
        labels = self.labels[sample_inds]
        return Data(sources, destination, timestamps, edge_idxs, labels)


def get_data_transductive(dataset_name, use_validation=True, dir="data"):
    assert dataset_name in ["Contacts", "lastfm", "wikipedia", "reddit", "askubuntu", "superuser", "wikitalk"]
    graph_df = pd.read_csv(f"../{dir}/{dataset_name}/ml_{dataset_name}.csv")

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values.astype(np.float64)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    n_edges = len(sources)

    random.seed(2024)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time

    val_mask = (
        np.logical_and(timestamps <= test_time, timestamps > val_time)
        if use_validation
        else test_mask
    )

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )

    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            full_data.n_interactions, full_data.n_unique_nodes
        )
    )
    print(
        "The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes
        )
    )
    print(
        "The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes
        )
    )
    print(
        "The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes
        )
    )

    return full_data, train_data, val_data, test_data, n_total_unique_nodes, n_edges


class MYNegEdgeSampler:
    def __init__(self,  destinations, full_destinations, num_neg, device='cuda', seed=2024):
        self.seed = seed
        self.destinations = destinations
        self.full_destinations = full_destinations
        self.num_neg = num_neg
        self.device = device

        assert num_neg <= len(full_destinations) - 1, f"num_neg should be <= {len(full_destinations) - 1}"

        self.unique_destinations = torch.tensor(np.unique(full_destinations), dtype=torch.long, device=self.device)
        self.num_dst_node = len(destinations)

        torch.manual_seed(self.seed)
    
    def sample(self, batch_destinations: np.ndarray):
        batch_size = len(batch_destinations)

        batch_destinations = torch.tensor(batch_destinations, dtype=torch.long, device=self.device)

        all_choices = self.unique_destinations.unsqueeze(0).expand(batch_size, -1)

        mask = (all_choices != batch_destinations.unsqueeze(1))

        all_choices = all_choices[mask].view(batch_size, -1) # [batch_size, len(self.unique_destinations)-1]

        random_scores = torch.rand_like(all_choices, dtype=torch.float, device=self.device)
        _, topk_indices = torch.topk(random_scores, self.num_neg, dim=1)

        batch_neg_samples = torch.gather(all_choices, 1, topk_indices)

        return batch_neg_samples
