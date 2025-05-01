# **EAGLE**

## 📦 Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

**Required packages and versions**:

- `matplotlib==3.8.4`
- `networkx==3.3`
- `numba==0.60.0`
- `numpy==1.26.4`
- `ogb==1.3.6`
- `pandas==1.5.3`
- `pyg-lib==0.4.0+pt20cu118`
- `scikit-learn==1.5.2`
- `scipy==1.13.1`
- `torch==2.0.1+cu118`
- `torch_geometric==2.5.3`

---

## 📊 Datasets

The datasets used for experiments can be found in the `Datasets` directory located at `../Datasets`. Please refer to the `Datasets/README.md` for more details on dataset access and usage.

**Note**: For the link prediction datasets, please place the 7 dataset folders downloaded from Google Drive under the `./data/` directory.

---

## ▶️ Run & Eval EAGLE

### EAGLE-Structure Example

To run EAGLE-Structure, use the following command:

```bash
python train_structure.py --dataset_name wikipedia --topk 100 --alpha 0.9 --beta 0.8 --gpu 0
```

The optimal parameters are as follows:

| Dataset     | $\alpha$ in Equation (8) | $\beta$ in Equation (9) | topk |
|-------------|-------|------|------|
| **Contacts**  | 0.3   | 0.3  | 100  |
| **LastFM**    | 0.9   | 0.2  | 100  |
| **Wikipedia** | 0.9   | 0.8  | 100   |
| **Reddit**    | 0.9   | 0.9  | 100  |
| **AskUbuntu** | 0.3   | 0.5  | 50   |
| **SuperUser** | 0.2   | 0.5  | 50   |
| **WikiTalk**  | 0.5   | 0.5  | 100  |

### EAGLE-Time Example

To run EAGLE-Time, use the following command:

```bash
python train_time.py --dataset_name wikipedia --topk 15 --lr 0.001 --weight_decay 5e-5 --gpu 0
```

The optimal parameters are as follows:

| Dataset     | *top-k*<sub>r</sub> in Equation (6)   | learning_rate | weight_decay |
|-------------|--------------------------------------|--------------------|-------------------|
| Contacts    | 30                                   | 0.0001             | 5e-05             |
| lastfm      | 20                                   | 0.0001             | 5e-05             |
| wikipedia   | 15                                   | 0.001              | 5e-05             |
| reddit      | 50                                   | 0.001              | 0.0               |
| askubuntu   | 30                                   | 0.0001             | 0.0               |
| superuser   | 30                                   | 0.0001             | 0.0               |
| wikitalk    | 30                                   | 0.001              | 0.0               |

### EAGLE-Hybrid Example

To run EAGLE-Hybrid, use the following command:

```bash
python train_hybrid.py --dataset_name wikipedia --gpu 0
```

**Note**: EAGLE-Hybrid is a weighted combination of EAGLE-Structure and EAGLE-Time. You need to first run structure module and time module with their optimal parameters listed above before training EAGLE-Hybrid.
