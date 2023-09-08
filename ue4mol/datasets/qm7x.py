import numpy as np
from typing import Callable, List, Optional
import h5py
import torch
from torch_geometric.data import Data, InMemoryDataset


class QM7X(InMemoryDataset):
    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 is_equilibrium: bool = True,
                 #full_dataset: bool = False
                 **kwargs
                 ):
        self.is_equilibrium = is_equilibrium
        if self.is_equilibrium == 'both':
            self.full_dataset = True
        else:
            self.full_dataset = False
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['1000.hdf5', '2000.hdf5', '3000.hdf5', '4000.hdf5', '5000.hdf5', '6000.hdf5', '7000.hdf5', '8000.hdf5']

    @property
    def processed_file_names(self) -> List[str]:
        if self.full_dataset:
            return ['full_data.pt']
        else:
            if self.is_equilibrium:
                return ['data.pt']
            else:
                return ['non_equilibrium_data.pt']
    
    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            fMOL = h5py.File(raw_path, 'r')
            mol_ids = list(fMOL.keys())
            for molid in mol_ids:
                ## get IDs of individual configurations/conformations of molecule
                conf_ids = list(fMOL[molid].keys())
                for i, conf_id in enumerate(conf_ids):
                    if self.full_dataset:
                        ## get atomic positions and numbers and add to molecules buffer
                        pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                        z = np.array(fMOL[molid][conf_id]['atNUM'])

                        ## get quantum mechanical properties and add them to properties buffer
                        energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                        force = np.array(fMOL[molid][conf_id]['totFOR'])
                    else:
                        if conf_id[-3:] != 'opt':
                            continue
                        if self.is_equilibrium:
                            ## get atomic positions and numbers and add to molecules buffer
                            pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                            z = np.array(fMOL[molid][conf_id]['atNUM'])

                            ## get quantum mechanical properties and add them to properties buffer
                            energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                            force = np.array(fMOL[molid][conf_id]['totFOR'])
                            if np.abs(force).min() == 0.0:
                                continue
                        else:
                            assert conf_ids[i-1][-3:] != 'opt'
                            ## get atomic positions and numbers and add to molecules buffer
                            pos = np.array(fMOL[molid][conf_ids[i-1]]['atXYZ'])
                            z = np.array(fMOL[molid][conf_ids[i-1]]['atNUM'])

                            ## get quantum mechanical properties and add them to properties buffer
                            energy = np.array(fMOL[molid][conf_ids[i-1]]['eMBD']) - np.array(fMOL[molid][conf_ids[i-1]]['eAT'])
                            force = np.array(fMOL[molid][conf_ids[i-1]]['totFOR'])

                        

                    z = torch.from_numpy(z).to(torch.long)
                    pos = torch.from_numpy(pos).to(torch.float)
                    energy = torch.from_numpy(energy).to(torch.float)
                    force = torch.from_numpy(force).to(torch.float)

                    data = Data(z=z, pos=pos, energy=energy, force=force)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"