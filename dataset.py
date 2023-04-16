import math
try:
    import ROOT as rt
except:
    print('WARNING: Cannot import ROOT')
import numpy as np
import os
import os.path as osp 
import torch
import itertools
import random
from torch_geometric.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data import IterableDataset
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import lmdb
import contextlib
import gzip
import importlib
import json
import io
import msgpack
import pickle as pkl
from pathlib import Path
import gc
import branch_config
tqdm = partial(tqdm, position=0, leave=True)

def deserialize(x, serialization_format):
    """
    Deserializes dataset `x` assuming format given by `serialization_format` (pkl, json, msgpack).
    """
    gc.disable()
    if serialization_format == 'pkl':
        return pkl.loads(x)
    elif serialization_format == 'json':
        serialized = json.loads(x)
    elif serialization_format == 'msgpack':
        serialized = msgpack.unpackb(x)
    else:
        raise RuntimeError('Invalid serialization format')
    gc.enable()
    return serialized

def serialize(x, serialization_format):
    """
    Serializes dataset `x` in format given by `serialization_format` (pkl, json, msgpack).
    """
    if serialization_format == 'pkl':
        # Pickle
        # Memory efficient but brittle across languages/python versions.
        return pkl.dumps(x)
    elif serialization_format == 'json':
        # JSON
        # Takes more memory, but widely supported.
        serialized = json.dumps(
            x, default=lambda df: json.loads(
                df.to_json(orient='split', double_precision=6))).encode()
    elif serialization_format == 'msgpack':
        # msgpack
        # A bit more memory efficient than json, a bit less supported.
        serialized = msgpack.packb(
            x, default=lambda df: df.to_dict(orient='split'))
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized

class TopRecoDataset(Dataset):
    def __init__(self, root_dir, n_top, cartesian, max_event=-1, set_num_raw_events=None, max_event_per_output=5000, num_output_files=-1, num_workers=1, raw_type='root', tree_name='output', transform=None, pre_transform=None, use_cache=True, debug=False, should_process=False):
        self.tree_name = tree_name
        self.should_process = should_process
        self.transform = transform
        self.max_event = max_event
        self.max_event_per_output = max_event_per_output
        self.num_output_files = num_output_files
        self.debug = debug
        self.cartesian = cartesian
        self.n_top = n_top
        self._root_dir = root_dir
        if cartesian:
            self._processed_dir = os.path.join(self._root_dir, 'processed', 'cartesian')
        else:
            self._processed_dir = os.path.join(self._root_dir, 'processed', 'detector')
        if not os.path.exists(self._processed_dir):
            os.makedirs(self._processed_dir)
        self.raw_type = raw_type
        self.num_workers = num_workers
        self._raw_dir = os.path.join(self._root_dir, 'raw')
        self.use_cache = use_cache
        self.cached_output = {}
        self.cached_num_raw_events = None
        self.cached_output = {}
        self.__indices__ = None
        if set_num_raw_events != None:
            print(f'INFO: Manually setting num_raw_events to {set_num_raw_events}')
            self.cached_num_raw_events = set_num_raw_events
        assert len(self.raw_file_names) <= 1, 'ERROR: expect at most 1 input file'
        
        # self.event_per_processed = event_per_processed
        # self.num_processed = math.ceil(self.num_raw_events / self.event_per_processed)
        # self.event_per_worker = math.ceil(self.event_per_processed / self.num_workers)
        # TTree branchnames

        self.branch_map = {}
        
        self.info_branches = []
        # standardization parameters
        self.node_shift = np.zeros(7)
        self.node_scale = np.ones(7)
        self.edge_shift = 0
        self.edge_scale = 1
        self.truth_top_shift = np.zeros(4)
        self.truth_top_scale = np.ones(4)


    @staticmethod    
    def mpi_to_pi(phi, debug=False):
        """ 
        Shift phi into the range of (-pi, pi)
        Inputs:
        - phi: A numpy array of arbitrary shape, with entries in (-2pi, 2pi)

        Returns:
        - a numpy array of the same shape as phi, but with each entries shifted into (-pi, pi)
        """
        assert np.sum((phi >= 2*np.pi) + (phi <= -2*np.pi)) == 0, 'ERROR: input phi is not in (-2pi, 2pi)'
        gt_pi = phi > np.pi
        lt_mpi = phi < -np.pi
        return phi - gt_pi * np.pi + lt_mpi * np.pi

    @staticmethod
    def to_cartesian(x):
        for i in range(len(x)):
            p = x[i, :4]
            v = rt.TLorentzVector()
            v.SetPtEtaPhiM(p[0], p[1], p[2], p[3])
            x[i, :4] = np.array([v.Px(), v.Py(), v.Pz(), v.E()])
        return x

    @staticmethod
    def to_detector(x):
        for i in range(len(x)):
            p = x[i, :4]
            v = rt.TLorentzVector()
            v.SetPxPyPzE(p[0], p[1], p[2], p[3])
            x[i, :4] = np.array([v.Pt(), v.Eta(), v.Phi(), v.M()])
        return x

    @staticmethod
    def convert_E_to_M(x):
        for i in range(len(x)):
            p = x[i, :4]
            v = rt.TLorentzVector()
            v.SetPtEtaPhiE(p[0], p[1], p[2], p[3])
            x[i, 3] = v.M()
        return x

    @property
    def num_raw_events(self):
        """ Total number of events in the raw input. If ROOT is not available, return the total number of events in the processed dataset. """
        if self.cached_num_raw_events:
            return self.cached_num_raw_events
        if self.raw_type == 'root':
            raw_file, tree = self.get_file_and_tree()
            num_evt = tree.GetEntries()
            raw_file.Close()
            self.cached_num_raw_events = num_evt
            return num_evt
        return -1

    @property
    def num_outputs(self):
        if self.num_output_files > 0:
            self.max_event_per_output = math.ceil(len(self) / self.num_output_files)
        print('Overriding max_event_per_output to ', self.max_event_per_output)
        return math.ceil(len(self) / self.max_event_per_output)
    
    def get_file_and_tree(self):
        raw_file = rt.TFile.Open(self.raw_file_name, 'READ')
        tree = raw_file.Get(self.tree_name)
        tree.SetBranchStatus('*', 0)
        for branch in list(self.branch_map.values()) + self.info_branches:
            tree.SetBranchStatus(branch, 1)
        return raw_file, tree

    def len(self):
        # number of processed events in this dataset
        return min(self.num_raw_events, self.max_event) if self.max_event >= 0 else self.num_raw_events

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self._raw_dir))

    @property
    def raw_file_name(self):
        return os.path.join(self._raw_dir, self.raw_file_names[0])

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.num_outputs)]

    def get(self, i):
        """ Return the i-th event """
        assert 0 <= i  < len(self), 'ERROR: event index out of range'
        output_index = i // self.max_event_per_output
        offset_in_output = i % self.max_event_per_output
        if output_index in self.cached_output:
            output_file = self.cached_output[output_index]
        else:
            output_file = torch.load(os.path.join(self._processed_dir, 'data_{}.pt'.format(output_index)))
            if self.use_cache:
                self.cached_output[output_index] = output_file
        try:
            data = output_file[offset_in_output]
        except:
            print(i, output_index, offset_in_output, len(output_file))
        return data

    def process(self, force_process=False):
        if force_process or self.should_process:
            start_index = 0
            end_index = len(self)
            output_indices = list(range(self.num_outputs))
            print(f'Number of outputs: {self.num_outputs}')
            with Pool(self.num_workers) as p:
                 process_in_range = partial(self.worker_process, start_index, end_index)
                 p.map(process_in_range, output_indices)


    def worker_process(self, start_index, end_index, output_index):
        worker_start = start_index + self.max_event_per_output * output_index
        worker_end = min(start_index + self.max_event_per_output * (output_index + 1), end_index)
        if self.raw_type == 'root':
            output_data = []
            file, tree = self.get_file_and_tree()
            for event_index in (range(worker_start, worker_end)):
                if self.debug:
                    print(f'DEBUG: Processing event {event_index}')
                event_data = self.create_data_from_event(tree, event_index)
                if event_data == None:
                    continue
                if self.transform != None:
                    event_data = self.transform(event_data)
                output_data.append(event_data)
            torch.save(output_data, os.path.join(self._processed_dir, 'data_{}.pt'.format(output_index)))
            file.Close()


    def create_data_from_event(self, tree, event_index):
        """ 
        Process an event into a graph, with additional information on truth matching and GNN predictions. 
        Use self.*shift and self.*scale to (partially) standarize the input and output.

        Inputs:
        - tree: A TTree
        - event_index: The entry number for the event in the TTree
        
        Returns a tuple of:
        - graph: A PyTorch Geometric data object, i.e, the input graph containing jets
        - truth_top: A numpy array of truth top 4-vectors, of shape (self.n_top, 4)
        - truth_matched: A numpy array indicating whether a top has a truth matched reco_triplet, of shape (self.n_top)
        - identified: A numpy array indicating whether a top has a truth matched reco_triplet and it is identified by the GNN, of shape (self.n_top)
        - reco_top: A numpy array of reco_triplet 4-vectors, containing -1 if a truth matched reco_triplet is not available, of shape (self.n_top, 4)
        - min_dR_candidate_top: A numpy array of GNN proposed candidate top 4-vectors, AFTER dR matching, containing -1 if DNE, of shape (self.n_top, 4)
        """
        raise NotImplementedError

class LMDBTopRecoDataset(TopRecoDataset):
    def worker_process(self, start_index, end_index, output_index):
        serialization_format = 'json'
        output_lmdb = os.path.join(self._processed_dir, 'data_{}'.format(output_index))
        worker_start = start_index + self.max_event_per_output * output_index
        worker_end = min(start_index + self.max_event_per_output * (output_index + 1), end_index)
        env = lmdb.open(str(output_lmdb), map_size=int(1e11))
        with env.begin(write=True) as txn:
            if self.raw_type == 'root':
                output_data = []
                file, tree = self.get_file_and_tree()
                i = 0
                for event_index in (range(worker_start, worker_end)):
                    if self.debug:
                        print(f'DEBUG: Processing event {event_index}')
                    event_data = self.create_data_from_event(tree, event_index)
                    if event_data == None:
                        continue
                    if self.transform != None:
                        event_data = self.transform(event_data)
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                        f.write(serialize(event_data, serialization_format))
                    compressed = buf.getvalue()
                    result = txn.put(str(i).encode(), compressed, overwrite=True)
                    if not result:
                        raise RuntimeError(f'Failed to write entry {i} in {str(output_lmdb)}')
                    i += 1
                txn.put(b'num_examples', str(i).encode())
                txn.put(b'serialization_format', serialization_format.encode())

class DelphesDataset(LMDBTopRecoDataset):
    def __init__(self, root_dir, n_top, cartesian, max_event=-1, set_num_raw_events=None, max_event_per_output=5000, num_workers=1, raw_type='root', transform=None, pre_transform=None, use_cache=True, debug=False, should_process=False):
        super().__init__(root_dir, n_top, cartesian, max_event, set_num_raw_events, max_event_per_output, num_workers, raw_type, transform, pre_transform, use_cache, debug, should_process)
        self.branch_map, self.info_branches = branch_config.branch_config_tuple['ttH_delphes']
    def create_data_from_event(self, tree, event_index):
        """ 
        Process an event into a graph, with additional information on truth matching and GNN predictions. 
        Use self.*shift and self.*scale to (partially) standarize the input and output.

        Inputs:
        - tree: A TTree
        - event_index: The entry number for the event in the TTree

        Returns a tuple of:
        - graph: A PyTorch Geometric data object, i.e, the input graph containing jets
        - truth_top: A numpy array of truth top 4-vectors, of shape (self.n_top, 4)
        - truth_matched: A numpy array indicating whether a top has a truth matched reco_triplet, of shape (self.n_top)
        - identified: A numpy array indicating whether a top has a truth matched reco_triplet and it is identified by the GNN, of shape (self.n_top)
        - reco_top: A numpy array of reco_triplet 4-vectors, containing -1 if a truth matched reco_triplet is not available, of shape (self.n_top, 4)
        - min_dR_candidate_top: A numpy array of GNN proposed candidate top 4-vectors, AFTER dR matching, containing -1 if DNE, of shape (self.n_top, 4)
        """
        tree.GetEntry(event_index)
        attr = {}
        for k, v in self.branch_map.items():
            attr[k] = getattr(tree, v)
        info = []
        for branch in  self.info_branches:
            info.append(getattr(tree, branch))

        # construct input nodes
        jet_btag = np.array([bool(tag) for tag in attr['jet_btag']])
        jet_features = np.array([attr['jet_pt'], attr['jet_eta'], attr['jet_phi'], attr['jet_m'], 1 - jet_btag, jet_btag, np.zeros_like(jet_btag)]).T # N_j * 7
        ph_features = np.array([attr['ph_pt'], attr['ph_eta'], attr['ph_phi'], np.zeros_like(attr['ph_phi']), np.zeros_like(attr['ph_phi']), np.zeros_like(attr['ph_phi']), np.ones_like(attr['ph_phi'])]).T # N_ph * 7
        node_features = np.concatenate([jet_features, ph_features], axis=0)
        n_node = node_features.shape[0]

        # calculate dR between objects
        dphi = np.expand_dims(node_features[:, 2], axis=0) - np.expand_dims(node_features[:, 2], axis=1)
        deta = np.expand_dims(node_features[:, 1], axis=0) - np.expand_dims(node_features[:, 1], axis=1)
        for idx in itertools.permutations(range(dphi.shape[0]), 2):
            dphi[idx] = self.mpi_to_pi(dphi[idx])
        dR_inv = 1/(np.sqrt(dphi ** 2 + deta ** 2) + 1e-5)

        # construct fully connected edges
        edges = list(itertools.permutations(range(n_node), 2))
        senders = np.array([x[0] for x in edges])
        receivers = np.array([x[1] for x in edges])
        edge_tuples = np.array([senders, receivers]) # 2 * num_edges
        n_edges = len(edges)
        edge_features = np.expand_dims(np.array([dR_inv[x] for x in edges]), axis=1) # num_edges * 1
        edge_features = edge_features.reshape(-1, 1) # num_edges * edge_dim

        # label: truth tops
        truth_top = np.array([attr['truth_top_pt'], attr['truth_top_eta'], attr['truth_top_phi'], attr['truth_top_m']]).T # self.N_top * 4
        if truth_top.shape[0] != self.n_top:
            print(f'WARNING: expect {self.n_top} truth tops, got {truth_top.shape[0]}, skipping')
            return None

        # For test events, compute additional non-input variables for truth matching and GNN prediction
        truth_matched = np.zeros(self.n_top)
        reco_top = -1 * np.ones_like(truth_top)
        for i in range(self.n_top):
            if not -1 == attr[f'reco_triplet_{i}_pt']:
                reco_top[i] = [attr[f'reco_triplet_{i}_pt'], attr[f'reco_triplet_{i}_eta'], attr[f'reco_triplet_{i}_phi'], attr[f'reco_triplet_{i}_m']]
                truth_matched[i] = 1
        identified = np.zeros(self.n_top)
        for i in range(self.n_top):
            reco_triplet = set(attr[f'reco_triplet_{i}'])
            if -1 in reco_triplet: # there is no truth matched reco_triplet
                continue
            else:
                for j in range(self.n_top): # since candidate triplets don't overlap, this does not reuse a candidate as desired
                    candidate_triplet = set([attr[f'top_candidate{j}_jet0'], attr[f'top_candidate{j}_jet1'], attr[f'top_candidate{j}_jet2']])
                    if reco_triplet == candidate_triplet:
                        identified[i] = 1

        # candidate_top: 4 vector of candidate top found by the GNN, in decreasing order of GNN score, NOT in the same order as truth tops
        candidate_top = -1 * np.ones_like(truth_top)
        for i in range(self.n_top):
            if not -1 == attr[f'top_candidate{i}_pt']:
                candidate_top[i] = [attr[f'top_candidate{i}_pt'], attr[f'top_candidate{i}_eta'], attr[f'top_candidate{i}_phi'], attr[f'top_candidate{i}_m']]
                if self.debug:
                    print(candidate_top)

        # compute dR between highest scoring candidate and truth tops
        dR00 = np.sqrt((truth_top[0, 1] - candidate_top[0, 1]) ** 2 + self.mpi_to_pi(truth_top[0, 2] - candidate_top[0, 2]) ** 2)
        dR10 = np.sqrt((truth_top[1, 1] - candidate_top[0, 1]) ** 2 + self.mpi_to_pi(truth_top[1, 2] - candidate_top[0, 2]) ** 2)

        # match highest scoring candidate to min dR truth top, which sorts the candidate tops in the order which represents as a backup prediction of the GNN 
        # we match in this way so that the highest scoring candidate takes priority during the match
        # the candidate top will not be used when identified is true, since we will just use reco_top directly.
        min_dR_candidate_top = np.copy(candidate_top)
        if dR10 < dR00:
            if self.debug:
                print('DEBUG: Swaping candidates during dR matching')
                print(candidate_top)
            min_dR_candidate_top[0], min_dR_candidate_top[1] = candidate_top[1], candidate_top[0]
            if self.debug:
                print('==>')
                print(min_dR_candidate_top)
        else:
            if self.debug:
                print('DEBUG: Not swaping candidates during dR matching')
        is_valid_candidate = 1 - (0 == np.sum(np.abs(min_dR_candidate_top - (-1) * np.ones([1, 4])), axis=1))
        if self.cartesian:
            node_features = self.to_cartesian(node_features)
            truth_top = self.to_cartesian(truth_top)
            reco_top = self.to_cartesian(reco_top)
            min_dR_candidate_top = self.to_cartesian(min_dR_candidate_top)

        return {
            'x': (node_features.tolist(), edge_tuples.tolist(), edge_features.tolist()),
            'y': truth_top.tolist(), 
            'truth_matched': truth_matched.tolist(), 
            'identified': identified.tolist(), 
            'reco_top': reco_top.tolist(), 
            'min_dR_candidate_top': min_dR_candidate_top.tolist(), 
            'is_valid_candidate': is_valid_candidate.tolist(),
            'info': info
        }

class ttHOfficialDataset(LMDBTopRecoDataset):
    def __init__(self, root_dir, n_top, cartesian, max_event=-1, set_num_raw_events=None, max_event_per_output=5000, num_workers=1, raw_type='root', transform=None, pre_transform=None, use_cache=True, debug=False, should_process=False):
        super().__init__(root_dir, n_top, cartesian, max_event, set_num_raw_events, max_event_per_output, num_workers, raw_type, transform, pre_transform, use_cache, debug, should_process)
        self.branch_map, self.info_branches = branch_config.branch_config_tuple['ttH_official']
    def create_data_from_event(self, tree, event_index):
        """ 
        Process an event into a graph, with additional information on truth matching and GNN predictions. 
        Use self.*shift and self.*scale to (partially) standarize the input and output.

        Inputs:
        - tree: A TTree
        - event_index: The entry number for the event in the TTree

        Returns a tuple of:
        - graph: A PyTorch Geometric data object, i.e, the input graph containing jets
        - truth_top: A numpy array of truth top 4-vectors, of shape (self.n_top, 4)
        - truth_matched: A numpy array indicating whether a top has a truth matched reco_triplet, of shape (self.n_top)
        - identified: A numpy array indicating whether a top has a truth matched reco_triplet and it is identified by the GNN, of shape (self.n_top)
        - reco_top: A numpy array of reco_triplet 4-vectors, containing -1 if a truth matched reco_triplet is not available, of shape (self.n_top, 4)
        - min_dR_candidate_top: A numpy array of GNN proposed candidate top 4-vectors, AFTER dR matching, containing -1 if DNE, of shape (self.n_top, 4)
        """
        tree.GetEntry(event_index)
        attr = {}
        for k, v in self.branch_map.items():
            attr[k] = getattr(tree, v)
        info = []
        for branch in  self.info_branches:
            info.append(getattr(tree, branch))

        # construct input nodes
        jet_btag = np.array([bool(tag) for tag in attr['jet_btag']])
        jet_features = np.array([attr['jet_pt'], attr['jet_eta'], attr['jet_phi'], attr['jet_m'], 1 - jet_btag, jet_btag, np.zeros_like(jet_btag)]).T # N_j * 7
        ph_features = np.array([[attr['ph_pt1'], attr['ph_eta1'], attr['ph_phi1'], 0, 0, 0, 1], 
                                [attr['ph_pt2'], attr['ph_eta2'], attr['ph_phi2'], 0, 0, 0, 1]]) # N_ph=2 * 7
        node_features = np.concatenate([jet_features, ph_features], axis=0).astype('float32')
        n_node = node_features.shape[0]

        # calculate dR between objects
        dphi = np.expand_dims(node_features[:, 2], axis=0) - np.expand_dims(node_features[:, 2], axis=1)
        deta = np.expand_dims(node_features[:, 1], axis=0) - np.expand_dims(node_features[:, 1], axis=1)
        for idx in itertools.permutations(range(dphi.shape[0]), 2):
            dphi[idx] = self.mpi_to_pi(dphi[idx])
        dR_inv = 1/(np.sqrt(dphi ** 2 + deta ** 2) + 1e-5)

        # construct fully connected edges
        edges = list(itertools.permutations(range(n_node), 2))
        senders = np.array([x[0] for x in edges])
        receivers = np.array([x[1] for x in edges])
        edge_tuples = np.array([senders, receivers]) # 2 * num_edges
        n_edges = len(edges)
        edge_features = np.expand_dims(np.array([dR_inv[x] for x in edges]), axis=1) # num_edges * 1
        edge_features = edge_features.reshape(-1, 1) # num_edges * edge_dim

        # label: truth tops
        truth_top = np.array([attr['truth_top_pt'], attr['truth_top_eta'], attr['truth_top_phi'], attr['truth_top_m']]).T # self.N_top * 4
        if truth_top.shape[0] != self.n_top:
            print(f'WARNING: expect {self.n_top} truth tops, got {truth_top.shape[0]}, skipping')
            return None


        # For test events, compute additional non-input variables for truth matching and GNN prediction
        truth_matched = np.zeros(self.n_top)
        reco_top = -1 * np.ones_like(truth_top)
        for i in range(self.n_top):
            reco_triplet_index = attr[f'reco_triplet_{i}']
            if not -1 in reco_triplet_index:
                v_jet = [rt.TLorentzVector(), rt.TLorentzVector(), rt.TLorentzVector()]
                for j in range(3):
                    v_jet[j].SetPtEtaPhiM(jet_features[reco_triplet_index[j], 0], jet_features[reco_triplet_index[j], 1], jet_features[reco_triplet_index[j], 2], jet_features[reco_triplet_index[j], 3])
                v_triplet = v_jet[0] + v_jet[1] + v_jet[2]
                reco_top[i] = [v_triplet.Pt(), v_triplet.Eta(), v_triplet.Phi(), v_triplet.M()]
                truth_matched[i] = 1
        identified = np.zeros(self.n_top)
        for i in range(self.n_top):
            reco_triplet = set(attr[f'reco_triplet_{i}'])
            if -1 in reco_triplet: # there is no truth matched reco_triplet
                continue
            else:
                for j in range(self.n_top): # since candidate triplets don't overlap, this does not reuse a candidate as desired
                    candidate_triplet = set([attr[f'top_candidate{j}_jet0'], attr[f'top_candidate{j}_jet1'], attr[f'top_candidate{j}_jet2']])
                    if reco_triplet == candidate_triplet:
                        identified[i] = 1

        # candidate_top: 4 vector of candidate top found by the GNN, in decreasing order of GNN score, NOT in the same order as truth tops
        candidate_top = -1 * np.ones_like(truth_top)
        for i in range(self.n_top):
            if not -1 == attr[f'top_candidate{i}_pt']:
                candidate_top[i] = [attr[f'top_candidate{i}_pt'], attr[f'top_candidate{i}_eta'], attr[f'top_candidate{i}_phi'], attr[f'top_candidate{i}_m']]
                if self.debug:
                    print(candidate_top)

        # compute dR between highest scoring candidate and truth tops
        dR00 = np.sqrt((truth_top[0, 1] - candidate_top[0, 1]) ** 2 + self.mpi_to_pi(truth_top[0, 2] - candidate_top[0, 2]) ** 2)
        dR10 = np.sqrt((truth_top[1, 1] - candidate_top[0, 1]) ** 2 + self.mpi_to_pi(truth_top[1, 2] - candidate_top[0, 2]) ** 2)

        # match highest scoring candidate to min dR truth top, which sorts the candidate tops in the order which represents as a backup prediction of the GNN 
        # we match in this way so that the highest scoring candidate takes priority during the match
        # the candidate top will not be used when identified is true, since we will just use reco_top directly.
        min_dR_candidate_top = np.copy(candidate_top)
        if dR10 < dR00:
            if self.debug:
                print('DEBUG: Swaping candidates during dR matching')
                print(candidate_top)
            min_dR_candidate_top[0], min_dR_candidate_top[1] = candidate_top[1], candidate_top[0]
            if self.debug:
                print('==>')
                print(min_dR_candidate_top)
        else:
            if self.debug:
                print('DEBUG: Not swaping candidates during dR matching')
        is_valid_candidate = 1 - (0 == np.sum(np.abs(min_dR_candidate_top - (-1) * np.ones([1, 4])), axis=1))
        if self.cartesian:
            node_features = self.to_cartesian(node_features)
            truth_top = self.to_cartesian(truth_top)
            reco_top = self.to_cartesian(reco_top)
            min_dR_candidate_top = self.to_cartesian(min_dR_candidate_top)

        return {
            'x': (node_features.tolist(), edge_tuples.tolist(), edge_features.tolist()),
            'y': truth_top.tolist(), 
            'truth_matched': truth_matched.tolist(), 
            'identified': identified.tolist(), 
            'reco_top': reco_top.tolist(), 
            'min_dR_candidate_top': min_dR_candidate_top.tolist(), 
            'is_valid_candidate': is_valid_candidate.tolist(),
            'info': info
        }

class tHDataset(LMDBTopRecoDataset):
    def __init__(self, root_dir, n_top, cartesian, max_event=-1, set_num_raw_events=None, max_event_per_output=5000, num_workers=1, raw_type='root', transform=None, pre_transform=None, use_cache=True, debug=False, should_process=False):
        super().__init__(root_dir, n_top, cartesian, max_event, set_num_raw_events, max_event_per_output, num_workers, raw_type, transform, pre_transform, use_cache, debug, should_process)
        self.branch_map, self.info_branches = branch_config.branch_config_tuple['tH']
    def create_data_from_event(self, tree, event_index):
        """ 
        Process an event into a graph, with additional information on truth matching and GNN predictions. 
        Use self.*shift and self.*scale to (partially) standarize the input and output.

        Inputs:
        - tree: A TTree
        - event_index: The entry number for the event in the TTree

        Returns a tuple of:
        - graph: A PyTorch Geometric data object, i.e, the input graph containing jets
        - truth_top: A numpy array of truth top 4-vectors, of shape (self.n_top, 4)
        - truth_matched: A numpy array indicating whether a top has a truth matched reco_triplet, of shape (self.n_top)
        - identified: A numpy array indicating whether a top has a truth matched reco_triplet and it is identified by the GNN, of shape (self.n_top)
        - reco_top: A numpy array of reco_triplet 4-vectors, containing -1 if a truth matched reco_triplet is not available, of shape (self.n_top, 4)
        - min_dR_candidate_top: A numpy array of GNN proposed candidate top 4-vectors, AFTER dR matching, containing -1 if DNE, of shape (self.n_top, 4)
        """
        tree.GetEntry(event_index)
        attr = {}
        for k, v in self.branch_map.items():
            attr[k] = getattr(tree, v)
        info = []
        for branch in  self.info_branches:
            info.append(getattr(tree, branch))

        # construct input nodes
        jet_btag = np.array([bool(tag) for tag in attr['jet_btag']])
        jet_features = np.array([attr['jet_pt'], attr['jet_eta'], attr['jet_phi'], attr['jet_m'], 1 - jet_btag, jet_btag, np.zeros_like(jet_btag)]).T # N_j * 7
        ph_features = np.array([[attr['ph_pt1'], attr['ph_eta1'], attr['ph_phi1'], 0, 0, 0, 1], 
                                [attr['ph_pt2'], attr['ph_eta2'], attr['ph_phi2'], 0, 0, 0, 1]]) # N_ph=2 * 7
        node_features = np.concatenate([jet_features, ph_features], axis=0).astype('float32')
        n_node = node_features.shape[0]

        # calculate dR between objects
        dphi = np.expand_dims(node_features[:, 2], axis=0) - np.expand_dims(node_features[:, 2], axis=1)
        deta = np.expand_dims(node_features[:, 1], axis=0) - np.expand_dims(node_features[:, 1], axis=1)
        for idx in itertools.permutations(range(dphi.shape[0]), 2):
            dphi[idx] = self.mpi_to_pi(dphi[idx])
        dR_inv = 1/(np.sqrt(dphi ** 2 + deta ** 2) + 1e-5)

        # construct fully connected edges
        edges = list(itertools.permutations(range(n_node), 2))
        senders = np.array([x[0] for x in edges])
        receivers = np.array([x[1] for x in edges])
        edge_tuples = np.array([senders, receivers]) # 2 * num_edges
        n_edges = len(edges)
        edge_features = np.expand_dims(np.array([dR_inv[x] for x in edges]), axis=1) # num_edges * 1
        edge_features = edge_features.reshape(-1, 1) # num_edges * edge_dim

        # label: truth tops
        truth_top = np.array([attr['truth_top_pt'], attr['truth_top_eta'], attr['truth_top_phi'], attr['truth_top_m']]).T # self.N_top * 4
        if truth_top.shape[0] != self.n_top:
            print(f'WARNING: expect {self.n_top} truth tops, got {truth_top.shape[0]}, skipping')
            return None


        # For test events, compute additional non-input variables for truth matching and GNN prediction
        truth_matched = np.zeros(self.n_top)
        reco_top = -1 * np.ones_like(truth_top)
        for i in range(self.n_top):
            reco_triplet_index = attr[f'reco_triplet_{i}']
            if not -1 in reco_triplet_index:
                v_jet = [rt.TLorentzVector(), rt.TLorentzVector(), rt.TLorentzVector()]
                for j in range(3):
                    v_jet[j].SetPtEtaPhiM(jet_features[reco_triplet_index[j], 0], jet_features[reco_triplet_index[j], 1], jet_features[reco_triplet_index[j], 2], jet_features[reco_triplet_index[j], 3])
                v_triplet = v_jet[0] + v_jet[1] + v_jet[2]
                reco_top[i] = [v_triplet.Pt(), v_triplet.Eta(), v_triplet.Phi(), v_triplet.M()]
                truth_matched[i] = 1
        identified = np.zeros(self.n_top)

        # candidate_top: 4 vector of candidate top found by the GNN, in decreasing order of GNN score, NOT in the same order as truth tops
        candidate_top = -1 * np.ones_like(truth_top)
        min_dR_candidate_top = np.copy(candidate_top)
        is_valid_candidate = np.zeros(self.n_top)
        if self.cartesian:
            node_features = self.to_cartesian(node_features)
            truth_top = self.to_cartesian(truth_top)
            reco_top = self.to_cartesian(reco_top)
            min_dR_candidate_top = self.to_cartesian(min_dR_candidate_top)

        return {
            'x': (node_features.tolist(), edge_tuples.tolist(), edge_features.tolist()),
            'y': truth_top.tolist(), 
            'truth_matched': truth_matched.tolist(), 
            'identified': identified.tolist(), 
            'reco_top': reco_top.tolist(), 
            'min_dR_candidate_top': min_dR_candidate_top.tolist(), 
            'is_valid_candidate': is_valid_candidate.tolist(),
            'info': info
        }


class XYData(Data):
    def __init__(self, x, y, x_edge_index, y_edge_index, xy_edge_index):
        super().__init__()
        self.x_in = x
        self.x_out = y
        self.edge_index_in = x_edge_index
        self.edge_index_out = y_edge_index
        self.edge_index_cross = xy_edge_index
    def __inc__(self, key, value):
        if 'edge_index_cross' == key:
            return torch.tensor([[self.x_in.size(0)], [self.x_out.size(0)]])
        elif 'edge_index_in' == key:
            return self.x_in.size(0)
        elif 'edge_index_out' == key:
            return self.x_out.size(0)
        else:
            return super().__inc__(key, value)

class VariableTopDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """

    def __init__(self, dataset, pad_y_up_to, test=False):
        """constructor
        """
        super().__init__()
        self.dataset = dataset
        self.edge_cache = {}
        self.test = test
        self.pad_y_up_to = pad_y_up_to

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        data = self.dataset[index]
        node_features, edge_tuples, _ = data['x']
        node_features = torch.FloatTensor(node_features)
        edge_tuples = torch.LongTensor(edge_tuples)
        graph = Data(node_features, edge_tuples)
        y = torch.FloatTensor(data['y'])
        num_target = y.shape[0]
        x = {'graph': graph, 'num_target': num_target}
        if not self.test:
            x['graph'] = self.get_xy_graph(graph, index, [num_target], self.pad_y_up_to)
        truth_matched = torch.LongTensor(data['truth_matched'])
        if 'identified' in data:
            identified = torch.LongTensor(data['identified'])
        else:
            identified = torch.zeros_like(truth_matched)
        if self.pad_y_up_to >= 0:
            y_pad = torch.zeros([self.pad_y_up_to - y.shape[0], y.shape[1]])
            truth_matched_pad = torch.zeros([self.pad_y_up_to - y.shape[0]])
            y = torch.cat([y, y_pad])
            truth_matched = torch.cat([truth_matched, truth_matched_pad])
            identified = torch.cat([identified, truth_matched_pad])
        if self.test:
            return {
                    'x': x, # input: graph, num_target
                    'y': y, # label: (padded) truth top kinematics
                    'num_target': num_target,
                    'truth_matched': truth_matched,
                    'identified': identified, 
                    'reco_top': torch.FloatTensor(data['reco_top']), 
                    # 'min_dR_candidate_top': torch.FloatTensor(data['min_dR_candidate_top']), 
                    # 'is_valid_candidate': torch.LongTensor(data['is_valid_candidate']), 
                    'info': torch.FloatTensor(data['info'])
            }
        else:
            return {
                    'x': x,
                    'y': y,
                    'num_target': num_target,
                    # 'truth_matched': truth_matched,
            }

    def get_xy_graph(self, input_graph, index, num_target, max_num_output, device=None):
        if not isinstance(input_graph, Batch):
            node_features, batch = input_graph.x, torch.zeros(input_graph.x.shape[0])
        else:
            node_features, batch = input_graph.x, input_graph.batch
        num_batches = int((torch.max(batch) + 1).item())
        if index in self.edge_cache:
            cross_attn_edges, self_attn_edges = self.edge_cache[index]
        else:
            # create cross attention edges between output nodes and input nodes
            source_nodes = torch.arange(batch.size(0), device=batch.device)
            cross_attn_edge_tensors = []
            for offset in range(max_num_output):
                output_nodes = max_num_output * batch + offset
                cross_attn_edge_tensors.append(torch.stack([source_nodes, output_nodes]))
            cross_attn_edges = torch.cat(cross_attn_edge_tensors, dim=1).long()
            # create self attention edges among output nodes
            self_attn_edge_tensors = [
                max_num_output * b +
                torch.stack([
                        i * torch.ones(num_target[b] - 1, device=batch.device), 
                        torch.cat([torch.arange(0, i, device=batch.device), torch.arange(i + 1, num_target[b], device=batch.device)])
                    ], dim=0)
                for b in range(num_batches) for i in range(num_target[b])
            ]
            self_attn_edges = torch.cat(self_attn_edge_tensors, dim=1).long()
            self.edge_cache[index] = (cross_attn_edges, self_attn_edges)
        if device == None:
            device = node_features.device
        return XYData(node_features, torch.zeros([num_batches * max_num_output, 4]).to(device), input_graph.edge_index, self_attn_edges.to(device), cross_attn_edges.to(device))

def add_output_edges(data, num_target): # actually we can just set max_num_output=num_target and make sure max_num_output >= num_target during training
    batch = torch.zeros(len(data['x'][0]))
    num_batches = 1
    # create cross attention edges between output nodes and input nodes
    source_nodes = torch.arange(batch.size(0))
    cross_attn_edge_tensors = []
    for offset in range(num_target):
        output_nodes = offset * torch.ones_like(batch)
        cross_attn_edge_tensors.append(torch.stack([source_nodes, output_nodes]))
    cross_attn_edges = torch.cat(cross_attn_edge_tensors, dim=1).long()
    # create self attention edges among output nodes
    self_attn_edge_tensors = [
        torch.stack([
                i * torch.ones(num_target - 1), 
                torch.cat([torch.arange(0, i), torch.arange(i + 1, num_target)])
            ], dim=0)
        for i in range(num_target)
    ]
    self_attn_edges = torch.cat(self_attn_edge_tensors, dim=1).long()
    data['output_edges'] = (cross_attn_edges.tolist(), self_attn_edges.tolist())
    return data


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, i):
        return self[i]

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if self._transform:
            item = self._transform(item)
        return item


class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """

    def __init__(self, data_file, transform=None, use_cache=False, readahead=False):
        """constructor
        """
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        env = lmdb.open(str(self.data_file), max_readers=100, readonly=True,
                        lock=False, readahead=readahead, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()

        self._env = env
        self._transform = transform
        self.cache = {}
        if use_cache:
            print('Using cache')
        self.use_cache = use_cache


    def __len__(self) -> int:
        return self._num_examples

    def get(self, i):
        return self[i]

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        if index in self.cache:
            return self.cache[index]
        with self._env.begin(write=False) as txn:
            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            item = deserialize(serialized, self._serialization_format)
        if self._transform:
            item = self._transform(item)
        if self.use_cache:
        	self.cache[index] = item
        return item

class IterableLMDBDataset(IterableDataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """

    def __init__(self, data_file, transform=None):
        """constructor
        """
        super().__init__()
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]
        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)
        env = lmdb.open(str(self.data_file), max_readers=100, readonly=True,
                        lock=False, readahead=True, meminit=False)
        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()
        self.start = 0
        self.end = self._num_examples
        self._env = env
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def len(self):
        return self._num_examples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_start = self.start
            worker_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_start = self.start + worker_id * per_worker
            worker_end = min(worker_start + per_worker, self.end)
        with self._env.begin(write=False) as txn:
            for index in range(worker_start, worker_end):
                compressed = txn.get(str(index).encode())
                buf = io.BytesIO(compressed)
                with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                    serialized = f.read()
                item = deserialize(serialized, self._serialization_format)
                if self._transform:
                    item = self._transform(item)
                yield item


class BufferedShuffleDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, buffer_size: int):
        super().__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.start = 0
        self.end = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        buf = []
        read_idx = 0
        for x in self.dataset:
            if read_idx < self.start:
                read_idx += 1
                continue
            elif read_idx >= self.end:
                return
            else:
                if len(buf) == self.buffer_size:
                    random.shuffle(buf)
                    while buf:
                        yield buf.pop()
                else:
                    buf.append(x)
                read_idx += 1
        random.shuffle(buf)
        while buf:
            yield buf.pop()


class MergedIterableDataset(IterableDataset):
    def __init__(self, datasets, sampling_frequencies, num_samples):
        super().__init__()
        assert len(datasets) == len(sampling_frequencies)
        self.datasets = datasets
        self.sampling_frequencies = np.array(sampling_frequencies) 
        self.sampling_frequencies = self.sampling_frequencies / np.sum(sampling_frequencies)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_samples = len(self)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_samples = len(self) // worker_info.num_workers
        iters = [iter(d) for d in self.datasets]
        for i in range(num_samples):
            dataset_index = np.random.choice(np.arange(len(self.datasets)), p=self.sampling_frequencies)
            try:
                data = next(iters[dataset_index])
                yield data
            except StopIteration:
                iters[dataset_index] = iter(self.datasets[dataset_index])
                data = next(iters[dataset_index])
                yield data

class FileDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """
    def __init__(self, data_file, transform=None):
        """constructor
        """
        super().__init__()
        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)
        self.data = torch.load(self.data_file)
        self._transform = transform
    def __len__(self) -> int:
        return len(self.data)

    def get(self, i):
        return self[i]

    def __getitem__(self, index: int):
        return self.data[i]




def make_lmdb_dataset(dataset, output_lmdb, filter_fn=None, serialization_format='json'):
    """
    Make an LMDB dataset from an input dataset.

    :param dataset: Input dataset to convert
    :type dataset: torch.utils.data.Dataset
    :param output_lmdb: Path to output LMDB.
    :type output_lmdb: Union[str, Path]
    :param filter_fn: Filter to decided if removing files.
    :type filter_fn: lambda x -> True/False
    :param serialization_format: How to serialize an entry.
    :type serialization_format: 'json', 'msgpack', 'pkl'
    :param include_bonds: Include bond information (only available for SDF yet).
    :type include_bonds: bool
    """

    num_examples = len(dataset)
    print(f'{num_examples} examples')
    env = lmdb.open(str(output_lmdb), map_size=int(1e11))

    with env.begin(write=True) as txn:
        try:
            i = 0
            for x in tqdm(dataset, total=num_examples):
                if filter_fn is not None and filter_fn(x):
                    continue
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                    f.write(serialize(x, serialization_format))
                compressed = buf.getvalue()
                result = txn.put(str(i).encode(), compressed, overwrite=True)
                if not result:
                    raise RuntimeError(f'LMDB entry {i} in {str(output_lmdb)} '
                                       'already exists')
                i += 1
        finally:
            txn.put(b'num_examples', str(i).encode())
            txn.put(b'serialization_format', serialization_format.encode())


class ListGraphDatset(Dataset):
    def __init__(self, data_file, relabel=lambda y: y):
      # dataset == (X, E, Y)
      X, E, Y = torch.load(data_file)
      self.X = X
      self.E = E
      self.Y = Y
      self.relabel = relabel
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        x = (self.X)[i]
        edge_index = (self.E)[i]
        y = self.relabel((self.Y)[i])
        graph = Data(x=x, edge_index=edge_index, y=y)
        return graph
