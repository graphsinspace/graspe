# specific instructions for torch 1.7.0/1.7.1 and no cuda
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
conda activate graspe

export CUDA=cpu
export TORCH=1.7.0

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
