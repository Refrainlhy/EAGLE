# Baseline: TGN

We use the code implementation from DyGLib benchmark ([DyGLib GitHub](https://github.com/yule-BUAA/DyGLib/)) for our experiments. 

According to the official paper ([TGN Paper](https://arxiv.org/abs/2006.10637)), the hyperparameter tuning ranges are as follows:

- **Number of most recent neighbors (k)**:  
  Number of neighbors sampled per layer for temporal graph aggregation, which controls the range of structural information a node can gather.  
  Tuned over `{5, 10, 20, 30, 40}` as noted in Section 5.2 (Figure 3b).

- **Number of attention layers (L)**:  
  Number of graph attention layers used in the embedding module at Section 3.1 - Embedding Block.  
  Tuned over `{1, 2}` as noted in Section 5.2 (Figure 3b).

- **Neighbor sampling strategy**:  
  Strategy to sample neighbors (`'last'` or `'uniform'`) in the embedding module. `'last'` selects the most recent neighbors, while `'uniform'` samples randomly.  
  Tuned over `{'last', 'uniform'}` as noted in Section 5.2 (Figure 3b) and Appendix A.4.1.

- **Dropout rate (dropout)**:  
  Probability used to randomly drop units in attention layers during training, helping prevent overfitting.  
  Tuned over `{0.1, 0.3, 0.5}`.

- **Memory dimension**:  
  Size of the vector `s_i(t)` used to store each node’s historical state at 3.1 - Memory Block.  
  Fixed to `172`  as listed in Appendix A.4.

- **Node embedding dimension**:  
  Dimension of the output node embedding `z_i(t)` used for downstream tasks at 3.1 - Memory Block.  
  Fixed to `100` according to Appendix A.4.

- **Time embedding dimension**:  
  Dimension of the time encoding vector φ(Δt) used in the temporal graph attention mechanism to represent time intervals at 3.1 - Memory Block.  
  Fixed to `100`, as specified in Appendix A.4.

- **Number of attention heads**:  
  Number of heads in the multi-head attention mechanism within the embedding module, allowing the model to attend to different representation subspaces at 3.1 - Memory Block.  
  Fixed to `2` for fair comparison as indicated in Appendix A.4.

- **Trainer**:  
  We conduct experiments using Adam optimizer with a learning rate of `0.0001` and weight decay of `0.0`. Training is performed for `100` epochs with early stopping based on the average precision (AP) metric, with a patience of `5` epochs. The trainer setting is consistent for all models to ensure a fair comparison.

The optimal hyperparameters for each dataset are as follows:

| **Dataset** | **num_nei_k** | **num_layer** | **nei_sam_strategy** | **dropout** |
|-------------|---------------|-----------------|----------------------|-------------|
| Wikipedia   | 10            | 1               | last                 | 0.1         |
| Reddit      | 10            | 1               | last                 | 0.3         |
| AskUbuntu   | 20            | 2               | last                 | 0.1         |
| SuperUser   | 20            | 2               | last                 | 0.1         |
| LastFM      | 20            | 2               | last                 | 0.1         |
