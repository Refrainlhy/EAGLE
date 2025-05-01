# Baseline: TGAT

According to the official paper ([TGAT Paper](https://arxiv.org/pdf/2002.07962)), the hyperparameter tuning ranges are as follows:


- **Number of TGAT Layers (num_layer)**:  
  Number of layers in the temporal graph attention network.  
  Tuned over `{1, 2, 3}` as noted in Section 4.4.

- **Number of Attention Head (num_head)**:  
  Number of parallel attention mechanisms.  
  Tuned over `{1, 2, 3, 4, 5}` as noted in Section 4.4.

- **Dropout rate (dropout)**:  
  Probability used to randomly drop units in attention layers during training, helping prevent overfitting.  
  Tuned over `{0.1, 0.3, 0.5}`.

- **Neighbors Dropout rate (nei_dropout)**:  
  Dropout rate determines how neighbors are sampled during aggregation for efficiency. 
  Tuned over `{0.0, 0.1, 0.3, 0.5}` as noted in Section 4.4 and add 0.0 for evaluating the best effectiveness.

- **Trainer**:  
  We conduct experiments using Adam optimizer with a learning rate of `0.0001` and weight decay of `0.0`. Training is performed for `100` epochs with early stopping based on the average precision (AP) metric, with a patience of `5` epochs. The trainer setting is consistent for all models to ensure a fair comparison.


The optimal hyperparameters for each dataset are as follows:

| **Dataset**  | **num_layer** | **num_head** | **dropout** | **nei_dropout** |
|--------------|---------------|--------------|-------------|-----------------|
| Wikipedia    | 2             | 2            | 0.1         | 0.0             |
| Reddit       | 2             | 2            | 0.2         | 0.0             |
| AskUbuntu    | 2             | 2            | 0.1         | 0.0             |
| SuperUser    | 2             | 3            | 0.1         | 0.0             |
| LastFM       | 2             | 2            | 0.1         | 0.0             |
| Contacts     | 2             | 2            | 0.1         | 0.0             |
| WikiTalk     | 2             | 2            | 32          | 0.0             |
