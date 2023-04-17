# download pretrained ttH model
echo "Downloading pretrained ttH model..."
model_dir="trained/CovariantTopFormer_(6, 6)_256_0901_ttH/"
mkdir -vp "${model_dir}"
mkdir -vp "${model_dir}/saved_models"
# download model weights
gdown 10z3FyIvyuVuo3zOFThTTGAb8R8H3g3rU -O "${model_dir}/saved_models/" 
# download precomputed inference output
gdown 1zSregpa-s8lgIIXucrg9Shkercdzguey -O "${model_dir}/" 
gdown 16PPIq7PyMfCdsKx4hy95TVlGWWrPXChc -O "${model_dir}/"
gdown 1-2KZURP0QmJOE2JuSMhJqYlKRqErgX4r -O "${model_dir}/"
gdown 1-JggFCwyteCiIb4R1NjgL4jras_QSRR9 -O "${model_dir}/"

# download pretrained tttt model
echo "Downloading pretrained tttt model..."
model_dir="trained/CovariantTopFormer_(6, 6)_256_0901_tttt/"
mkdir -vp "${model_dir}"
mkdir -vp "${model_dir}/saved_models"
# download model weights
gdown 18Hll5O-uboumzIrPzwyZ1YP6K4MiQ_-_ -O "${model_dir}/saved_models/"
# download precomputed inference output
gdown 1-BEX1iylRKsvui4vytzinguXCSONkRRG -O "${model_dir}/"
