# MiniBiomeGPT — Public research demo

This application runs **real inference** from the MiniBiomeGPT T2D classifier saved by the original Colab notebook. It is not a medical device or diagnostic tool.

## How to publish

1. Run `MiniBiomeGPT_Export_for_Public_Demo.ipynb` in Google Colab. It reads the original Drive folder `MyDrive/MiniBiomeGPT_T2D` and creates `MiniBiomeGPT_Ready_To_Deploy.zip` with genuine weights and feature order.
2. Unzip the exported archive. **Upload its contents** (not the ZIP file) to a GitHub repository. The repository should have `app.py`, `model.py`, `requirements.txt`, `models/`, and `data/` at the root.
3. Open https://share.streamlit.io and choose *Create app → Yup, I have an app*. Choose the repository, branch `main`, and entry point `app.py`.
4. Deploy. Verify that the public site loads and that **Run MiniBiomeGPT** returns a score.

## Safety and scientific notes

- The example inputs are **group means** calculated only from the original model-training subset (not actual patients). They are for software demonstration, not scientific evaluation.
- Only a trained checkpoint, feature names, example group averages and recorded metrics are published. No raw human sample files, subject identifiers or clinical metadata are included.
- Scores are exploratory and not clinically calibrated. Do not use the app for medical decisions.
- The reported random forest external-test AUROC 0.69 is from Karlsson. The saved transformer validation AUROC 0.5678 is from Qin. These are **not matched evaluations**.
- Check the original data source's licensing terms and whether public redistribution of model weights and aggregated values is permitted before publication.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Technical provenance

The model architecture and rank-abundance binning match the supplied `MiniBiomeGPT_T2D.ipynb` notebook (architecture cell 34; preprocessing cell 33; saved classifier checkpoint cell 36). The model has 64-dimensional embeddings, four heads and two transformer encoder layers. The exported checkpoint records 289 features.
