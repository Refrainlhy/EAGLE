# Baseline: Zebra

We use the code implementation from the official repository Zebra provided by the authors ([Zebra GitHub](https://github.com/LuckyLYM/Zebra/)) for our experiments. 

According to the official paper ([Zebra Paper](https://www.vldb.org/pvldb/vol16/p1332-li.pdf)), the hyperparameter tuning ranges are as follows:


- **$\alpha$ list**:  
  Controls the temporal weighting of the diffusion process; a smaller $\alpha$ captures influences over longer temporal spans, while a larger $\alpha$ focuses more on recent neighbors​.  
  Tuned over `{0.1, 0.2, 0.3, 0.4, 0.5}`.

- **$\beta$ list**:  
  Modifies the exponential decay of the temporal influence in T-PPR, where a smaller $\beta$ focuses on short-term interactions, and a larger $\beta$ considers longer-range temporal influences​.  
  Tuned over `{0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}`.

- **TPPR strategy**:  
  Strategy to control how the T-PPR calculation is carried out at Section 6.2. `'streaming'` indicates obtaining approximate values through stream computing, while `'pruning'` reduces the computational load through pruning.  
  Tuned over `{'streaming', 'pruning'}`.

- **Number of most recent neighbors (n_degree)**:  
  Number of neighbors sampled for per node, which controls the range of structural information a node can gather.  
  Tuned over `{10, 20, 30}`.

- **Dropout rate (dropout)**:  
  Probability used to randomly drop units in attention layers during training, helping prevent overfitting.  
  Tuned over `{0.1, 0.2, 0.3}`.

- **Trainer**:  
  We conduct experiments using Adam optimizer with a learning rate of `0.0001` and weight decay of `0.0`. Training is performed for `100` epochs with early stopping based on the average precision (AP) metric, with a patience of `5` epochs. The trainer setting is consistent for all models to ensure a fair comparison.

The optimal hyperparameters for each dataset are as follows:

| Dataset       | $\alpha$ list         | $\beta$ list            | tppr_strategy | n_degree | dropout |
|---------------|--------------------|----------------------|---------------|----------|---------|
| **Wikipedia** | [0.1]              | [0.5, 0.95]          | streaming     | 10       | 0.1     |
| **Reddit**    | [0.1]              | [0.5, 0.95]          | streaming     | 20       | 0.1     |
| **AskUbuntu** | [0.1, 0.2]        | [0.5, 0.7]           | pruning       | 10       | 0.3     |
| **SuperUser** | [0.1, 0.2]        | [0.5, 0.7]           | pruning       | 10       | 0.3     |
| **LastFM**    | [0.1]         | [0.7, 0.9]           | pruning       | 15       | 0.2     |
| **Contacts**  | [0.1]         | [0.7, 0.9]           | pruning       | 15       | 0.2     |
| **WikiTalk**  | [0.1]              | [0.8, 0.95]          | streaming     | 10       | 0.2     |
