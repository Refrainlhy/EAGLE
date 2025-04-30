# Baseline: GraphMixer

We use the code implementation from DyGLib benchmark ([DyGLib GitHub](https://github.com/yule-BUAA/DyGLib/)) for our experiments. 

According to the official paper ([GraphMixer Paper](https://openreview.net/pdf?id=ayPPc0SyLv1)), the hyperparameter tuning ranges are as follows:

- **Time-slot size (T)**:  
  Defines the temporal window for collecting neighbors in the node encoder. Fixed at `2000` as noted in Section A.3.

- **Number of 1-hop most recent neighbors (K)**:  
  Specifies how many recent 1-hop temporal neighbors are used for each node, corresponding to `num_neighbors` in DyGLib. Tuned over `{10, 20, 30}`.

- **Number of transformer layers (num_layer)**:  
  Controls the depth of the MLP-mixer module. Tuned over `{1, 2, 3}`.

- **Dropout rate (dropout)**:  
  Regulates regularization strength during training. Tuned over `{0.1, 0.3, 0.5}`.

- **Trainer**:  
  We conduct experiments using Adam optimizer with a learning rate of `0.0001` and weight decay of `0.0`. Training is performed for `100` epochs with early stopping based on the average precision (AP) metric, with a patience of `5` epochs. The trainer setting is consistent for all models to ensure a fair comparison.


The optimal hyperparameters for each dataset are as follows:

| **Dataset**    | **num_neighbors_K** | **num_layer** | **dropout** |
|----------------|----------------------|---------------|-------------|
| Wikipedia      | 30                   | 2             | 0.3         |
| Reddit         | 10                   | 2             | 0.5         |
| AskUbuntu      | 20                   | 2             | 0.3         |
