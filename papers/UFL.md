# Summary of UFL 

[original paper](https://arxiv.org/abs/1908.02877)

- Goal: developed an unsupervised learning approach to discriminate between individual instances, completely ignoring class labels and allowing the network to learn similarity between instances without a need for semantic categories.

- Motivation: Transfer learning, [UFL](https://arxiv.org/abs/1805.01978) method has great generalization ability
- Approach:
    1. Training
        1. feature extractor: any SOTA CNN backbone
        2. similarity search: cosine similarity + noise-contrastive estimation
        3. trained using a Memory bank, loss func: NLL
    
    2. Inference: similarity_search_func(query_image)
       - feature vector + clf (weighted KNN)
    
- Experiment: [Xview Data](http://xviewdataset.org/#dataset) object detection task using RGB imagery

- Conclusion: feature vectors from UFL could do amazing things:
    - similarity search
    - outlier detection
    - hierarchical clustering
    - ... definitely synthetic and real data analysis! 

# Xview data properties
1. small bbox, low resolution
2. rotations/shifts of bbox --> problem
3. extreme class imbalance --> use Focal loss 
4. labeling errors --> room for improvements

# Key Take-aways

- When cropping bbox:
    - use square chips without reshaping which includes neighbor context information, which agrees to our previous methods;
    - more complex methods gave NO measurable benefits
    
- Hidden space dim:
    - 128 works fine, different values in feature dim do NOT offer significant benefits.
    
- Trick: employ random augmentation (flips, rotations, color gitter etc.) for better generalization ability.

- Do not train the auto-encoder from scratch, use a pre-trained feature extracotr on ImageNet as baseline! 

- Try pretrained UFL using ResNet50 instead of auto-encoder in our work! 

- Possible directions: can we adapt the similarity search idea to our latent space method?
    - project synthetic data onto real space
    - find top-k similar synthetic images given a real image. What can we learn from there?
    
- Visualization in reports: Use [T-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) and [UMAP](https://arxiv.org/abs/1802.03426) when presenting feature vectors. 

