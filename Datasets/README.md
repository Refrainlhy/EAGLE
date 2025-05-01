# 📊 Dataset

## 🔗 Link Prediction

We conducted experiments on the `Contacts`, `LastFM`, `Wikipedia`, `Reddit`, `AskUbuntu`, `SuperUser`, and `WikiTalk` datasets.
The raw dataset files can be obtained from the following link:

[Google Drive Link](https://drive.google.com/drive/folders/1pV3sKQyK-N5wvinwo6l7DuM26-hPBTPO?usp=drive_link)

Note: For our experimental setup, we generated 99 negative samples for each sample in the test set. For details on data processing, please refer to `link_data_loader.py`.

## 🎯 Node Classification

We conducted experiments on the `Trade`, `Genre`, `Reddit`, and `Token` datasets, using the raw dataset files and splits provided by [TGB Datasets](https://tgb.complexdatalab.com/docs/nodeprop/).

To install the TGB dependency, run:

```bash
pip install py-tgb
```

We recommend using the TGB's interface, which includes downloading and standardized splitting. You can refer to `node_data_loader.py` for example usage of this interface.

For further details, please consult the [TGB documentation](https://docs.tgb.complexdatalab.com/).
