from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import os
import numpy as np
from sklearn.model_selection import train_test_split


class FaceScrubCropped(Dataset):
    def __init__(
        self, train, group='all', transform=None, cropped=True, root='data/facescrub', download=False, seed: int = 42
    ):
        """
        group: 'actors', 'actresses', 'all
        train: True, False
        """

        if group == 'actors':
            if cropped:
                root = os.path.join(root, 'actors/faces')
            else:
                root = os.path.join(root, 'actors/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
        elif group == 'actresses':
            if cropped:
                root = os.path.join(root, 'actresses/faces')
            else:
                root = os.path.join(root, 'actresses/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
        elif group == 'all':
            if cropped:
                root_actors = os.path.join(root, 'actors/faces')
                root_actresses = os.path.join(root, 'actresses/faces')
            else:
                root_actors = os.path.join(root, 'actors/images')
                root_actresses = os.path.join(root, 'actresses/images')
            dataset_actors = ImageFolder(root=root_actors, transform=None)
            target_transform_actresses = lambda x: x + len(dataset_actors.classes)
            dataset_actresses = ImageFolder(
                root=root_actresses, transform=None, target_transform=target_transform_actresses
            )
            dataset_actresses.class_to_idx = {
                key: value + len(dataset_actors.classes)
                for key, value in dataset_actresses.class_to_idx.items()
            }
            self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
            self.classes = dataset_actors.classes + dataset_actresses.classes

            # fix the names of the actors in the classes list
            index_freddy_prinze_jr = self.classes.index('Freddy_Prinze_Jr.')
            self.classes[index_freddy_prinze_jr] = 'Freddy_Prinze_Jr'
            # fix typo in Leslie Nielsen
            index_leslie_nielsen = self.classes.index('Leslie_Neilsen')
            self.classes[index_leslie_nielsen] = 'Leslie_Nielsen'

            # fix typo in Robert De Niro
            index_robert_de_niro = self.classes.index('Robert_Di_Niro')
            self.classes[index_robert_de_niro] = 'Robert_De_Niro'

            # remove middle name from Tatyana Ali
            index_tatyana_ali = self.classes.index('Tatyana_M._Ali')
            self.classes[index_tatyana_ali] = 'Tatyana_Ali'

            self.class_to_idx = {**dataset_actors.class_to_idx, **dataset_actresses.class_to_idx}
            self.targets = dataset_actors.targets + [t + len(dataset_actors.classes) for t in dataset_actresses.targets]
        else:
            raise ValueError(f'Dataset group {group} not found. Valid arguments are \'actors\' and \'actresses\'.')

        self.transform = transform

        # split the dataset into a train and a test set using the indices
        indices = list(range(len(self.dataset)))
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=seed, stratify=self.targets)

        if train:
            self.dataset = Subset(self.dataset, train_idx)
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = Subset(self.dataset, test_idx)
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]

        if self.transform is not None:
            im = self.transform(im)

        return im, self.targets[idx]
