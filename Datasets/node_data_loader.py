from torch_geometric.loader import TemporalDataLoader
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

def load_and_process_dataset(dataset_name="tgbn-trade", root="datasets", val_ratio=0.15, test_ratio=0.15, batch_size=200):
    """
    dataset_name: Dataset name ('tgbn-xxx')
    root: Root directory for data storage
    val_ratio: Validation set ratio (default: 0.15)
    test_ratio: Test set ratio (default: 0.15)
    batch_size: Batch size for DataLoader (fixed for fair comparison: 200)
    """
    dataset = PyGNodePropPredDataset(name=dataset_name, root=root)

    num_classes = dataset.num_classes
    eval_metric = dataset.eval_metric
    
    print(f"Name: {dataset_name}")
    print(f"#Class: {num_classes}\nMetric: {eval_metric}")

    temporal_data = dataset.get_TemporalData()
    
    # Split
    train_data, val_data, test_data = temporal_data.train_val_test_split(
        val_ratio=val_ratio, 
        test_ratio=test_ratio
    )

    print(f"#Training Sample: {len(train_data)}")
    print(f"#Validation Sample: {len(val_data)}")
    print(f"#Test Sample: {len(test_data)}")

    train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
    val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
    test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

    return dataset, temporal_data, (train_loader, val_loader, test_loader)

if __name__ == "__main__":
    dataset, temporal_data, loaders = load_and_process_dataset()

    train_loader, val_loader, test_loader = loaders

    label_time = dataset.get_label_time()
    dataset.reset_label_time()
