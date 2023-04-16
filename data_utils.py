################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network data loading utilities
################################################################################


import torch
import os.path as osp

from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch

import h5py
import json

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

class PDBBindDataset(Dataset):
    def __init__(
        self,
        data_file,
        dataset_name,
        feature_type,
        preprocessing_type,
        use_docking=False,
        output_info=False,
        cache_data=True,
        h5_file_driver=None,
        distance_cutoff=5.,
        ligand_only=False
    ):
        super().__init__()
        self.dataset_name = dataset_name

        self.data_file = data_file
        self.feature_type = feature_type
        self.preprocessing_type = preprocessing_type
        self.use_docking = use_docking
        self.output_info = output_info
        self.cache_data = cache_data
        self.data_dict = {}  # will use this to store data once it has been computed if cache_data is True

        self.data_list = []  # will use this to store ids for data

        self.h5_file_driver = h5_file_driver
        self.distance_cutoff = distance_cutoff
        self.ligand_only = ligand_only


        if self.use_docking:

            with h5py.File(data_file, "r") as f:

                for name in list(f):
                    # if the feature type (pybel or rdkit) not available, skip over it
                    if self.feature_type in list(f[name]):
                        affinity = np.asarray(f[name].attrs["affinity"]).reshape(1, -1)
                        if self.preprocessing_type in f[name][self.feature_type]:
                            if self.dataset_name in list(
                                f[name][self.feature_type][self.preprocessing_type]
                            ):
                                for pose in f[name][self.feature_type][
                                    self.preprocessing_type
                                ][self.dataset_name]:
                                    self.data_list.append((name, pose, affinity))

        else:

            with h5py.File(data_file, "r", driver=self.h5_file_driver) as f:

                for name in list(f):
                    # if the feature type (pybel or rdkit) not available, skip over it
                    if self.feature_type in list(f[name]):
                        affinity = np.asarray(f[name].attrs["affinity"]).reshape(1, -1)

                        self.data_list.append(
                            (name, 0, affinity)
                        )  # Putting 0 for pose to denote experimental structure and to be consistent with docking data format

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        if self.cache_data:

            if item in self.data_dict.keys():
                return self.data_dict[item]

            else:
                pass       
 
        
        pdbid, pose, affinity = self.data_list[item]

        node_feats, coords = None, None
        with h5py.File(self.data_file, "r") as f:

            if (
                not self.dataset_name
                in f[
                    "{}/{}/{}".format(
                        pdbid, self.feature_type, self.preprocessing_type
                    )
                ].keys()
            ):
                print(pdbid)
                return None

            if self.use_docking:
                # TODO: the next line will cuase runtime error because not selelcting poses
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ][pose]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )
                    ][pose]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            else:
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )

                    ]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            else:
                raise NotImplementedError

        # account for the vdw radii in distance cacluations (consider each atom as a sphere, distance between spheres)

        if self.ligand_only:
            ligand_mask = node_feats[:, 14] == 1
            node_feats = node_feats[ligand_mask]
            coords = coords[ligand_mask]
        dists = pairwise_distances(coords, metric="euclidean")

        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
        mask = edge_attr < self.distance_cutoff
        edge_index = edge_index[:, mask.view(-1)]

        x = torch.from_numpy(node_feats).float()

        y = torch.FloatTensor(affinity).view(-1, 1)
        data = Data(
            x=x, edge_index=edge_index, y=y
        )
        data.coords = torch.FloatTensor(coords)


        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, pose, data)

            else:
                self.data_dict[item] = data

            return self.data_dict[item]

        else:
            if self.output_info:
                return (pdbid, pose, data)
            else:
                return data


class DockedPDBBindDataset(Dataset):
    def __init__(
        self,
        data_file,
        cache_data=True,
        h5_file_driver=None,
        distance_cutoff=5.,
    ):
        super().__init__()

        self.data_file = data_file
        self.cache_data = cache_data
        self.data_dict = {}  # will use this to store data once it has been computed if cache_data is True

        self.data_list = []  # will use this to store ids for data

        self.h5_file_driver = h5_file_driver
        self.distance_cutoff = distance_cutoff

        with open(data_file.replace('.h5', '.json')) as f:
            f_ = json.load(f)
            names = list(f_['regression'].keys())
            self.data_list = names
        correct_file = '/content/drive/MyDrive/LBA/data/pdbbindv2019_pocket_rmsd_1.json'
        with open(correct_file) as f:
            f_ = json.load(f)
            names = list(f_.keys())
            self.correct_list = names

        self.file = h5py.File(self.data_file, "r")
        print('Target is now affinity!')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                return self.data_dict[item]
            else:
                pass
        
        name = self.data_list[item]    
        node_feats = torch.FloatTensor(self.file['regression'][name]['pdbbind_sgcnn']['node_feats'])
        
        affinity = torch.FloatTensor(self.file['regression'][name].attrs["affinity"]).reshape(1, -1)
        dock_score = torch.FloatTensor(self.file['regression'][name].attrs["dock_score"]).reshape(1, -1)
        correct = torch.BoolTensor([name in self.correct_list]).reshape(1, -1)
       
        coords = torch.FloatTensor(self.file['regression'][name]['pdbbind_sgcnn']['dists'])
        dists = pairwise_distances(coords, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
        mask = edge_attr < self.distance_cutoff
        edge_index = edge_index[:, mask.view(-1)]

        x = node_feats
        data = Data(
            x=x, coords=coords, edge_index=edge_index, y=affinity, correct=correct, name=name
        )

        if self.cache_data:
            self.data_dict[item] = data
            return self.data_dict[item]
        else:
            return data

class NoisyPDBBindDataset(PDBBindDataset):
    def __init__(
        self,
        data_file,
        dataset_name,
        feature_type,
        preprocessing_type,
        use_docking=False,
        output_info=False,
        cache_data=True,
        h5_file_driver=None,
        sigma=0,
    ):
        super().__init__(
            data_file,
            dataset_name,
            feature_type,
            preprocessing_type,
            use_docking,
            output_info,
            cache_data,
            h5_file_driver,
        )
        self.sigma = sigma

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                if self.output_info:
                    pdbid, pose, coords, x, y = self.data_dict[item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape), a_min=-2*self.sigma, a_max=2*self.sigma)
                    dists = pairwise_distances(coords + eps, metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
                    data = Data(
                        x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y
                    )
                    return (pdbid, pose, data)
                else:
                    coords, x, y = self.data_dict[item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape), a_min=-2*self.sigma, a_max=2*self.sigma)
                    dists = pairwise_distances(coords + eps, metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
                    x = torch.from_numpy(node_feats).float()
                    y = torch.FloatTensor(affinity).view(-1, 1)
                    data = Data(
                        x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y
                    )
                    return data
            else:
                pass       
 
        
        pdbid, pose, affinity = self.data_list[item]

        node_feats, coords = None, None
        with h5py.File(self.data_file, "r") as f:

            if (
                not self.dataset_name
                in f[
                    "{}/{}/{}".format(
                        pdbid, self.feature_type, self.preprocessing_type
                    )
                ].keys()
            ):
                print(pdbid)
                return None

            if self.use_docking:
                # TODO: the next line will cuase runtime error because not selelcting poses
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ][pose]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )
                    ][pose]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            else:
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )

                    ]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            else:
                raise NotImplementedError

        # account for the vdw radii in distance cacluations (consider each atom as a sphere, distance between spheres)
        eps = np.clip(self.sigma * np.random.randn(*coords.shape), a_min=-2*self.sigma, a_max=2*self.sigma)
        dists = pairwise_distances(coords + eps, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y
        )


        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, pose, coords, x, y)
                return (pdbid, pose, data)
            else:
                self.data_dict[item] = (coords, x, y)
                return data
        else:
            if self.output_info:
                return (pdbid, pose, data)
            else:
                return data

class EdgeNoisePDBBindDataset(PDBBindDataset):
    def __init__(
        self,
        data_file,
        dataset_name,
        feature_type,
        preprocessing_type,
        use_docking=False,
        output_info=False,
        cache_data=True,
        h5_file_driver=None,
        sigma=0,
    ):
        super().__init__(
            data_file,
            dataset_name,
            feature_type,
            preprocessing_type,
            use_docking,
            output_info,
            cache_data,
            h5_file_driver,
        )
        self.sigma = sigma

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                if self.output_info:
                    pdbid, pose, coords, x, y, edge_attr0 = self.data_dict[item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape), a_min=-2*self.sigma, a_max=2*self.sigma)
                    dists = pairwise_distances(coords + eps, metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
                    edge_attr0 = edge_attr0.view(-1, 1)
                    edge_attr = edge_attr.view(-1, 1)
                    if not edge_attr0.shape == edge_attr.shape: #, f'{dists0.shape} != {dists.shape} or {edge_attr0.shape} != {edge_attr.shape}'
                        print(f'Wrong shape for edge_index, skipping {pdbid}')
                        return self[item + 1]
                    edge_attr_combined = torch.cat([edge_attr0, edge_attr], dim=1)
                    data = Data(
                        x=x, edge_index=edge_index, edge_attr=edge_attr_combined.view(-1, 2), y=y
                    )
                    return (pdbid, pose, data)
                else:
                    coords, x, y, edge_attr0 = self.data_dict[item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape), a_min=-2*self.sigma, a_max=2*self.sigma)
                    dists = pairwise_distances(coords + eps, metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
                    edge_attr0 = edge_attr0.view(-1, 1)
                    edge_attr = edge_attr.view(-1, 1)
                    if not edge_attr0.shape == edge_attr.shape: #, f'{dists0.shape} != {dists.shape} or {edge_attr0.shape} != {edge_attr.shape}'
                        print(f'Wrong shape for edge_index, skipping {pdbid}')
                        return self[item + 1]
                    edge_attr_combined = torch.cat([edge_attr0, edge_attr], dim=1)
                    data = Data(
                        x=x, edge_index=edge_index, edge_attr=edge_attr_combined.view(-1, 2), y=y
                    )
                    return data
            else:
                pass       
 
        
        pdbid, pose, affinity = self.data_list[item]

        node_feats, coords = None, None
        with h5py.File(self.data_file, "r") as f:

            if (
                not self.dataset_name
                in f[
                    "{}/{}/{}".format(
                        pdbid, self.feature_type, self.preprocessing_type
                    )
                ].keys()
            ):
                print(pdbid)
                return None

            if self.use_docking:
                # TODO: the next line will cuase runtime error because not selelcting poses
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ][pose]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )
                    ][pose]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            else:
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )

                    ]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            else:
                raise NotImplementedError

        # account for the vdw radii in distance cacluations (consider each atom as a sphere, distance between spheres)
        dists0 = pairwise_distances(coords, metric="euclidean")
        edge_index, edge_attr0 = dense_to_sparse(torch.from_numpy(dists0).float())
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        eps = np.clip(self.sigma * np.random.randn(*coords.shape), a_min=-2*self.sigma, a_max=2*self.sigma)
        dists = pairwise_distances(coords + eps, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
        edge_attr0 = edge_attr0.view(-1, 1)
        edge_attr = edge_attr.view(-1, 1)
        if not edge_attr0.shape == edge_attr.shape: #, f'{dists0.shape} != {dists.shape} or {edge_attr0.shape} != {edge_attr.shape}'
            print(f'Wrong shape for edge_index, skipping {pdbid}')
            return self[item + 1]
        edge_attr_combined = torch.cat([edge_attr0, edge_attr], dim=1)
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr_combined.view(-1, 2), y=y
        )

        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, pose, coords, x, y, edge_attr0)
                return (pdbid, pose, data)
            else:
                self.data_dict[item] = (coords, x, y, edge_attr0)
                return data
        else:
            if self.output_info:
                return (pdbid, pose, data)
            else:
                return data  