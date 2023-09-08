import numpy as np
import os.path as osp
import getpass

import torch
from torch_geometric.datasets import QM9, MD17
from torch_geometric.data import DataLoader as GeomDataLoader
import torch_geometric.transforms as T
from .transformations import *
from .qm7x import QM7X

transformation = {
    "pos_rnd_translation": lambda translate: T.RandomTranslate(translate=translate),
    "pos_rnd_rotation": lambda degrees: T.RandomRotate(degrees=degrees),
    "pos_rnd_scaling": lambda scales: T.RandomScale(scales=scales),
    "pos_rnd_noise": lambda noise_magnitude: PositionRandomNoise(noise_magnitude=noise_magnitude),
    "ft_rnd_scaling": lambda scales: FeatureRandomScale(scales=scales),
    "ft_rnd_noise": lambda noise_magnitude: FeatureRandomNoise(noise_magnitude=noise_magnitude),
    # "edge_rnd_noise": lambda scales: T.RandomScale(scales=scales),
}

def get_dataloader(
        dataset_name,
        target,        
        batch_size,
        val_batch_size=None,
        transforms=[],
        transform_params=[],
        split=[.8, .1, .1],
        n_test_data=None,
        seed=42,
        dataset_directory=None,
        num_workers=6,
        debug=False,
        normalizing=True,
        num_datapoints=None,
        create_eps_env=False,
        eps_env_params={},
        **kwargs):
    """
    Return train, validation and test data loader for a dataset given batching and splitting parameters.

    :param dataset_directory: string. Directory where the dataset is/will be saved.
    :param dataset_name: string. Name of the dataset.
    :param batch_size: int. Size of the training batch.
    :param batch_size_eval: int. Size of the evaluation batch.
    :param split: list of 2 ints. Determine the split between train and validation and validation and test sets.
    :param n_test_data: int (Default: None). If not None, specify the maximum size of the test set.
    :param seed: int. Random seed for the data split.
    :return: train_loader, val_loader, test_loader
    """

    if dataset_directory is None:
        dataset_directory = '/nfs/staff-hdd/' + getpass.getuser() + '/datasets'
        
    path = osp.join(dataset_directory, dataset_name)

    transform_composition = []
    if transforms is not None:
        for transform, transform_param in zip(transforms, transform_params):
            transform_composition.append(transformation[transform](transform_param))
    transform_composition = T.Compose(transform_composition)
    print(transform_composition)
    print(path)

    mean = 0
    std = 1
    DataLoader = GeomDataLoader
    if dataset_name == 'QM7X':
        num_train = 32000
        num_val = 5000
        dataset = QM7X(path, is_equilibrium=kwargs['is_equilibrium'])
        indices = list(range(len(dataset)))
        assert np.sum(split) == 1.0
        np.random.seed(seed)
        np.random.shuffle(indices)
        # dataset_rng = torch.Generator()
        # dataset_rng.manual_seed(seed)
        # indices = torch.randperm(len(dataset), generator=dataset_rng).numpy() 
        split_0, split_1 = int(len(dataset) * split[0]), int(len(dataset) * split[1])
        split_1 += split_0
        train_indices, val_indices, test_indices = indices[:split_0], indices[split_0:split_1], indices[split_1:]
        train_indices = train_indices[:num_train]
        val_indices = val_indices[:num_val]

        if normalizing:
            mean = torch.mean(dataset.data.energy[train_indices])
            std = torch.std(dataset.data.energy[train_indices])
            dataset.data.energy = (dataset.data.energy - mean)
        train_dataset = dataset.index_select(train_indices)
        val_dataset = dataset.index_select(val_indices)
        test_dataset = dataset.index_select(test_indices)


        
    elif dataset_name == 'QM9':
        dataset = QM9(path, transform=transform_composition)
        # DimeNet uses the atomization energy for targets U0, U, H, and G.
        idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
        dataset.data.y = dataset.data.y[:, idx]


        target_dim = qm9_target_map(target)
        dataset.data.y = dataset.data.y[:, target_dim]
        indices = list(range(len(dataset)))
        assert np.sum(split) == 1.0
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_0, split_1 = int(len(dataset) * split[0]), int(len(dataset) * split[1])
        split_1 += split_0
        train_indices, val_indices, test_indices = indices[:split_0], indices[split_0:split_1], indices[split_1:]
        if normalizing:
            mean = torch.mean(dataset.data.y[train_indices])
            std = torch.std(dataset.data.y[train_indices])
            dataset.data.y = (dataset.data.y - mean) / std
        train_dataset = dataset.index_select(train_indices)
        val_dataset = dataset.index_select(val_indices)
        test_dataset = dataset.index_select(test_indices)
        
    elif dataset_name == 'MD17':

        dataset = MD17(path, target, transform=transform_composition)
        indices = list(range(len(dataset)))
        assert np.sum(split) == 1.0
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_0, split_1 = int(len(dataset) * split[0]), int(len(dataset) * split[1])
        split_1 += split_0
        train_indices, val_indices, test_indices = indices[:split_0], indices[split_0:split_1], indices[split_1:]
        if num_datapoints:
            train_indices = train_indices[:num_datapoints]
            val_indices = val_indices[:num_datapoints]
            print(f'using only {num_datapoints} datapoints')

        if normalizing:
            mean = torch.mean(dataset.data.energy[train_indices])
            std = torch.std(dataset.data.energy[train_indices])
            #dataset.data.energy = (dataset.data.energy - mean) / std
            print("only subtracting mean")
            dataset.data.energy = dataset.data.energy - mean
            #dataset.data.force = dataset.data.force / std
        train_dataset = dataset.index_select(train_indices)
        val_dataset = dataset.index_select(val_indices)
        test_dataset = dataset.index_select(test_indices)
    else:
        raise NotImplementedError

    if n_test_data is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, np.arange(min(n_test_data, len(test_dataset))))

    if not val_batch_size:
        val_batch_size = batch_size
    
    print(f'normalizing with -- mean={mean} and std={std}')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        create_eps_env=create_eps_env, 
        eps_env_params=eps_env_params
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        create_eps_env=create_eps_env,
        eps_env_params=eps_env_params
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        create_eps_env=create_eps_env,
        eps_env_params=eps_env_params
    )
    return train_loader, val_loader, test_loader, {'mean': mean, 'std': std}


def qm9_target_map(target):
    targets = {
      'mu': 0,
      'alpha': 1,
      'homo': 2,
      'lumo': 3,
      'gap': 4, 
      'r2': 5,
      'zpve': 6,
      'U0': 7,
      'U': 8,
      'H': 9,
      'G': 10,
      'Cv': 11,
      'U0_atom': 12
    }
    assert target in targets, f"the target is not found. it has to be one of {targets.keys()}"
    return targets[target]
