# setup environment and install dependencies
conda create --name cpt python=3.9
conda activate cpt
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip3 uninstall torch-scatter torch-sparse
pip3 install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip3 install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip3 install torch-geometric==1.7.2
pip3 install lmdb
pip3 install matplotlib
pip3 install seaborn
pip3 install wandb
pip3 install gdown
conda deactivate