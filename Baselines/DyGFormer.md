# Baseline: DyGFormer

We use the code implementation from DyGLib benchmark ([DyGLib GitHub](https://github.com/yule-BUAA/DyGLib/)) for our experiments. 

We thank them for providing a standardized and user-friendly benchmark!

According to the official paper ([DyGFormer Paper](https://arxiv.org/abs/2303.13047)), the hyperparameter tuning ranges are as follows:


- **Number of transformer layers (num_layer)**:  
  Number of transformer encoder layers used at Section 4.1 - Transformer Encoder.  
  Tuned over `{1, 2}`.

- **Number of most recent neighbors (num_neighbors)**:  
  Number of historical neighbors sampled used at Section 4.1 - Neighbor Co-occurrence Encoding Scheme, which controls the range of structural information a node can gather.  
  Tuned over `{10, 20, 30}`.

- **Number of transformer layers (patch_size)**:  
  Number of multiple non-overlapping patches $P$ that the encoding sequence are divided into at Section 4.1 - Patching Technique.  
  Tuned over `{1, 2, 4, 8, 16}`.

- **​​Input sequence length (max_input_sequence_length)**:  
  Pad length $\left| \mathcal{S}^{u}_{t} \right|$ at Section 4.1 - Patching Technique.  
  Tuned over `{32, 64, 128, 256, 512}`.

- **Dropout rate (dropout)**:  
  Probability used to randomly drop units in attention layers during training, helping prevent overfitting.  
  Tuned over `{0.1, 0.2, 0.3, 0.4, 0.5}`.

- **Time embedding dimension (d_T)**:  
  Dimension of time feature interval encodingat at Section 4.1 - Encoding Neighbors, Links, and Time Intervals.  
  Fixed to `100` as noted in Section 5.1.

- **Number of attention heads**:  
  Number of heads in the multi-head attention mechanism used at Section 4.1 - Transformer Encoder.  
  Fixed to `2` as noted in Section 5.1.

- **Trainer**:  
  We conduct experiments using Adam optimizer with a learning rate of `0.0001` and weight decay of `0.0`. Training is performed for `100` epochs with early stopping based on the average precision (AP) metric, with a patience of `5` epochs. The trainer setting is consistent for all models to ensure a fair comparison.


The optimal hyperparameters for each dataset are as follows:

| Dataset      | num_layer | num_neighbors | patch_size | max_input_sequence_length | dropout |
|--------------|-----------|---------------|------------|----------------------------|---------|
| Wikipedia    | 2         | 20            | 4          | 128                        | 0.1     |
| Reddit       | 1         | 10            | 1          | 32                         | 0.3     |
| AskUbuntu    | 2         | 20            | 2          | 64                         | 0.2     |
