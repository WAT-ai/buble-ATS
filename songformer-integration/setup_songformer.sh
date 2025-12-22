# Setup script for SongFormer
echo "Setting up SongFormer..."
cd ../open-source-models/songformer

# Check if conda environment exists
if conda env list | grep -q "songformer"; then
    echo "✓ SongFormer environment already exists"
else
    echo "Creating SongFormer conda environment..."
    conda create -n songformer python=3.10 -y
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate songformer

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download models if not present
if [ ! -f "src/SongFormer/ckpts/SongFormer.safetensors" ]; then
    echo "Downloading pretrained models..."
    cd src/SongFormer
    python utils/fetch_pretrained.py
    cd ../..
fi

echo "✓ SongFormer setup complete!"
echo "Activate with: conda activate songformer"
