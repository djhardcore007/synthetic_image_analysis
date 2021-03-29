# README #

This folder contains the code and output of UMAP visualization.

### What is the clustering folder for? ###

The goal of clustering is to compare the synthetic and real data in the latent space.

If we could observe some overlap between the real and synthetic feature maps in the latent space, we might conclude that the synthetic data shares the same underlying distribution as the real data. In general, the more overlap, the better.

The visualization folder contains the real and synthetic images level 2 to 6. Mapped from real to synthetic data and the other way around.

<img src="visualization/rareplane/3d_UMAP_p2 (Synthetic mapped to Real latent space).png" alt="3d_UMAP_p2 (Synthetic mapped to Real latent space)" style="zoom:30%;" />
<img src="visualization/rareplane/3d_UMAP_p2 (Real mapped to Synthetic latent space).png" alt="3d_UMAP_p2 (Real mapped to Synthetic latent space)" style="zoom:30%;" />

### Deployment instructions ###

Please open a Google Colab with GPU setting.

First, download the libraries.

```bash
# install dependencies: (use cu101 because colab has CUDA 10.1)
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab
```

```bash
# install detectron2
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

```bash
# install hdbscan
!pip install hdbscan
```

Then, sample 100 images from both real and synthetic data. Crop images. 
Store them in data_dir as 'real_images.npy' and 'syn_images.npy'. Make sure np arr shape = (100, 512, 512 ,3).

Finally, run the script.

```bash
# Get feature maps for 5 different layers of FPN 
!python3 predict_backbone.py --model_dir ... --output_dir ... --data_dir ...

# Get UMAP visualization
!python3 makeUmap.py --n_samples 100 --output_dir ... --data_dir ...
```

### Who do I talk to? ###

* Repo owner or admin: Florence Jiang
* Other community or team contact: Lucy Wang