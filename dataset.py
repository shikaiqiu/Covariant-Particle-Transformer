from torch_geometric.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data import IterableDataset
from functools import partial
from tqdm import tqdm
import lmdb
import gzip
import json
import io
import pickle as pkl
from pathlib import Path
import gc
tqdm = partial(tqdm, position=0, leave=True)

def deserialize(x, serialization_format):
    gc.disable()
    if serialization_format == 'json':
        serialized = json.loads(x)
    else:
        raise RuntimeError('Invalid serialization format')
    gc.enable()
    return serialized

def serialize(x, serialization_format):
    if serialization_format == 'json':
        serialized = json.dumps(
            x, default=lambda df: json.loads(
                df.to_json(orient='split', double_precision=6))).encode()
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized


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