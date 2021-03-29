# UMAP Visualization result

This folder contains umap visualization for both Rareplane and Xview data. Please see more detailed model and data information in sub-directories.

### What is the UMAP Pipeline? ###

1. Sample 100 synthetic and real images each from full datasets.

2. Crop real images into (512, 512, 3) such that synthetic images and real images are of the same size. Now syn or real data shape is (100, 512, 512, 3)

3. Use the backbone of a pretrained detectron2 model (we get this pretrained model from our customers) to get feature maps for 5 different Feauture Pyramid Networks.

   - p2 (100, 16384) for only real or synthetic data
   - p3 (100, 4096)
   - p4 (100, 1024)
   - p5 (100, 256)
   - p6 (100, 64)

   The output is saved in '.npy files'

4. For each pyramid level, we vertically stack the real and synthetic data to get the following shape:

   - p2 (200, 16384) for both real and synthetic data
   - p3 (200, 4096)
   - p4 (200, 1024)
   - p5 (200, 256)
   - p6 (200, 64)

5. Use UMAP to reduce feautres. We fit a UMAP model using only real data (synthetic data), and then transform the real data (synthetic data) using that model. Intuitively, we have mapped the synehtic data (real data) onto a latent space of real data (synthetic data).

   - p2 (200, 16384) 	--> (200,3)
   - p3 (200, 4096)	--> (200,3)
   - p4 (200, 1024)	--> (200,3)
   - p5 (200, 256)		--> (200,3)
   - p6 (200, 64)		--> (200,3)

6. Visualization using 3D scatter plot.