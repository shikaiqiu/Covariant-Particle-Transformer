echo "NOTE: sometimes 'gdown' can fail, if so please download the datasets manually from link printed in the error message and resume the script."
mkdir -vp data/final

# ttH
id=1-9w64szj-hcElSFLl6zedoyy8j10ZpfB
gdown $id -O data/final/ds.zip
unzip data/final/ds.zip
rm data/final/ds.zip

# ttbar
id=1-38_q4uJgq517g4XW2-t_csLQetX35mM
gdown $id -O data/final/ds.zip
unzip data/final/ds.zip
rm data/final/ds.zip

# ttW
id=1-3jCkU6r2Fg4iyc116grvF6HrY3Ixoba
gdown $id -O data/final/ds.zip
unzip data/final/ds.zip
rm data/final/ds.zip

# ttH_odd
id=1-79sOYpXNyMbsGgMD2fgZ9rVp5owWBvr
gdown $id -O data/final/ds.zip
unzip data/final/ds.zip
rm data/final/ds.zip

# tttt
id=1-hTvGmhgXP_pCUHKUEMUhd2IpV5pKL8M
gdown $id -O data/final/ds.zip
unzip data/final/ds.zip
rm data/final/ds.zip
# for this dataset we download the number.pt file separately
id=1AFH8iNe0VB3WQr-I7zyE_4oOt7W2NPEU
gdown $id -O data/final/tttt/number.pt