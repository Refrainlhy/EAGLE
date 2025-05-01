# Baseline: JODIE

According to the official paper ([JODIE Paper](https://arxiv.org/abs/1908.01207)), the hyperparameter tuning ranges are as follows:

- **Dynamic embedding size (emb_dim)**:  
  Dimensionality of the embeddings that are updated over time through the sequence of interactions. Tuned over `{32, 64, 128, 256}` as noted in Section 4.5.

- **Number of attention layers (num_layer)**:  
  Number of graph attention layers used in the embedding module at Section 3.1 - Embedding Block.  
  Tuned over `{1, 2}`.

- **Dropout rate (dropout)**:  
  Probability used to randomly drop units in attention layers during training, helping prevent overfitting.  
  Tuned over `{0.1, 0.3, 0.5}`.

- **Trainer**:  
  We conduct experiments using Adam optimizer with a learning rate of `0.0001` and weight decay of `0.0`. Training is performed for `100` epochs with early stopping based on the average precision (AP) metric, with a patience of `5` epochs. The trainer setting is consistent for all models to ensure a fair comparison.


The optimal hyperparameters for each dataset are as follows:

| **Dataset** | **emb_dim** | **num_layer** | **dropout** |
|-------------|-------------|---------------|-------------|
| Wikipedia   | 128         | 2             | 0.1         |
| Reddit      | 128         | 1             | 0.1         |
