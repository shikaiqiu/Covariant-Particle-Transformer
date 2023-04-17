echo "NOTE: sometimes 'gdown' can fail, if so please download the datasets manually from link printed in the error message and resume the script."
mkdir -vp data/final
# download ttH dataset
id=1-9w64szj-hcElSFLl6zedoyy8j10ZpfB
gdown $id -O data/final/ds.zip
unzip data/final/ds.zip -d data/final/ttH
rm data/final/ds.zip
