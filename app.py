"""Public scientific demonstration, live inference using exported MiniBiomeGPT weights."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model import load_checkpoint, predict_probability

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "minibiomegpt_t2d.pt"
FEATURE_PATH = ROOT / "data" / "selected_species.json"
PROFILE_PATH = ROOT / "data" / "aggregate_examples.csv"
PRETRAIN_PATH = ROOT / "models" / "minibiomegpt_pretrained.pt"
METRICS_PATH = ROOT / "data" / "experiment_results.csv"

st.set_page_config(page_title="MiniBiomeGPT | Research Demo", page_icon="🧬", layout="wide")

st.title("🧬 MiniBiomeGPT")
st.subheader("Microbiome representation learning — interactive research demonstration")
st.info("**Research prototype only.** The model predicts a T2D-associated label from shotgun-derived species profiles. It is not a diagnostic tool. Example profiles are TRAINING GROUP AGGREGATES, not individual patients. They are intended only to illustrate model operation.")

missing = [str(p.relative_to(ROOT)) for p in (MODEL_PATH, FEATURE_PATH, PROFILE_PATH) if not p.is_file()]
if missing:
    st.error("Model export incomplete. Missing files: " + ", ".join(missing))
    st.markdown("Run `MiniBiomeGPT_Export_for_Public_Demo.ipynb` in Google Colab. It builds the full deployment ZIP using your saved checkpoint.")
    st.stop()

@st.cache_resource(show_spinner="Loading the original trained model on CPU …")
def get_model():
    features = json.loads(FEATURE_PATH.read_text(encoding="utf-8"))
    model, checkpoint = load_checkpoint(MODEL_PATH, features)
    return features, model, checkpoint

try:
    features, model, checkpoint = get_model()
except Exception as exc:
    st.error(f"Could not load the trained checkpoint: {exc}")
    st.stop()

examples = pd.read_csv(PROFILE_PATH)
if "profile_name" not in examples.columns or len(examples) < 1:
    st.error("The example profiles file is malformed")
    st.stop()
if not all(col in examples.columns for col in features):
    st.error("The example file does not include every trained species column")
    st.stop()

st.caption(f"Trained transformer · {len(features)} species features · two encoder layers · four attention heads · CPU inference")

tab1, tab2, tab3, tab4 = st.tabs([
    "Run the model", "Explore species", "Research benchmarks", "Methods & limitations"
])

with tab1:
    st.markdown("### Run **actual trained-model inference**")
    st.write("Choose an aggregated reference profile and optionally modify the species values. Every prediction calls the trained MiniBiomeGPT classifier, not a saved prediction table.")
    names = examples["profile_name"].astype(str).tolist()
    choice = st.selectbox("Choose an illustrative reference profile", names)
    selected_row = examples.loc[examples["profile_name"] == choice].iloc[0]
    base_profile = pd.to_numeric(selected_row[features], errors="coerce")
    if base_profile.isna().any() or (base_profile < 0).any():
        st.error("The selected example contains invalid species abundances")
        st.stop()
    vector = base_profile.to_numpy(dtype=np.float64).copy()
    st.caption("Values are *percentage-like relative-abundance measurements*. The original model converts them into per-sample rank bins; it does not directly use raw percentages.")

    editable_species = base_profile.sort_values(ascending=False).head(20).index.tolist()
    editor = pd.DataFrame({
        "Species": [x.removeprefix("species:") for x in editable_species],
        "Relative abundance (%)": [float(base_profile[x]) for x in editable_species]
    })
    edited = st.data_editor(
        editor, key=f"species_editor_{choice}", hide_index=True,
        disabled=["Species"], use_container_width=True,
        column_config={"Relative abundance (%)": st.column_config.NumberColumn(min_value=0., max_value=100., format="%.4f")}
    )
    changes = pd.to_numeric(edited["Relative abundance (%)"], errors="coerce")
    if changes.isna().any() or (changes < 0).any():
        st.error("Only finite nonnegative abundances are allowed")
        st.stop()
    species_index = {name: i for i, name in enumerate(features)}
    for column, abundance in zip(editable_species, changes):
        vector[species_index[column]] = float(abundance)

    if st.button("Run MiniBiomeGPT", type="primary"):
        try:
            prob, bins = predict_probability(model, vector)
            c1, c2, c3 = st.columns(3)
            c1.metric("T2D-class score", f"{prob:.3f}")
            c2.metric("Illustrative class (0.5 threshold)", "T2D" if prob >= .5 else "Healthy")
            c3.metric("Detected model species", int(np.sum(bins > 0)))
            st.warning("This is a model score for an edited or aggregated profile, not a calibrated clinical probability, diagnosis, or estimate of disease risk.")
            st.caption("The input has been converted with the original five-rank-bin transformation. All 289 input features remain in the original training order.")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

    st.divider()
    st.markdown("### Test your own compatible sample")
    st.write("The CSV must contain one row and **all model species columns** with exactly matching names. Missing columns are rejected instead of silently filled with zeros.")
    template = pd.DataFrame([dict(zip(features, vector))])
    st.download_button("Download a correctly formatted single-sample CSV", template.to_csv(index=False).encode("utf-8"), "minibiomegpt_input_example.csv", "text/csv")
    uploaded = st.file_uploader("Upload a compatible one-row CSV (optional)", type=["csv"], max_upload_size=2)
    if uploaded is not None:
        try:
            sample = pd.read_csv(uploaded)
            if len(sample) != 1:
                raise ValueError("Please upload exactly one row")
            missing_features = [x for x in features if x not in sample.columns]
            if missing_features:
                raise ValueError(f"Missing {len(missing_features)} columns. First missing: {missing_features[:3]}")
            values = sample.loc[0, features].apply(pd.to_numeric, errors="coerce")
            if values.isna().any():
                raise ValueError("All model abundance columns must contain valid numbers")
            st.success("Sample format validated. Values are processed in memory and are not saved by this app.")
            if st.button("Run uploaded sample"):
                score, _ = predict_probability(model, values.to_numpy(dtype=float))
                st.metric("Exploratory T2D-class score", f"{score:.3f}")
                st.warning("Not for medical interpretation. The model was not established as a diagnostic classifier.")
        except Exception as exc:
            st.error(f"Invalid sample: {exc}")

with tab2:
    st.markdown("### Species-abundance explorer")
    show_n = st.slider("Number of highest-abundance species", 5, 40, 20)
    profile_df = pd.DataFrame({"Species": [f.removeprefix("species:") for f in features], "Relative abundance (%)": vector})
    top = profile_df.sort_values("Relative abundance (%)", ascending=False).head(show_n)
    st.bar_chart(top.set_index("Species"), horizontal=True)
    st.dataframe(top, hide_index=True, use_container_width=True)
    st.caption("Displayed values come from group-average profiles assembled from the training portion of the original notebook, with optional edits in the first tab. They are not representative of an individual microbiome.")

with tab3:
    st.markdown("### Results recorded in the original research notebook")
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics, hide_index=True, use_container_width=True)
    else:
        st.warning("Benchmark file not present in this deployment")
    st.markdown("**Important:** The random forest external-test result (Karlsson, 96 samples) and the transformer internal validation result (Qin, 363 samples) used **different evaluation cohorts**. Their AUROCs are not a direct head-to-head comparison.")
    if PRETRAIN_PATH.exists():
        try:
            import torch
            pretrained = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=True)
            losses = pretrained.get("pretrain_losses", [])
            if losses:
                st.markdown("#### Masked-species pretraining history")
                st.line_chart(pd.DataFrame({"Masked-species MSE": losses}, index=np.arange(1,len(losses)+1)))
        except Exception as exc:
            st.caption(f"Pretraining curve unavailable: {exc}")
    st.caption("Historical benchmark values were reported by the research notebook. This page does not re-run cohort evaluation. The live inference tab does execute the exported trained checkpoint.")

with tab4:
    st.markdown("### Scientific method")
    st.write("Human stool shotgun-metagenomic species profiles were obtained through curatedMetagenomicData. The original notebook harmonised Healthy and T2D labels, excluded the HMP repeated-subject cohort, and selected species using training-data prevalence only.")
    st.write("MiniBiomeGPT represents each species with a learned species embedding and an abundance embedding. Two transformer encoder layers process the community. A CLS output supports binary disease classification. Self-supervised pretraining masks detected species values and predicts their abundance ranks.")
    st.markdown("**Train/validation/test design:** transformer training used 963 samples from non-Qin/non-Karlsson studies; QinJ_2012 provided 363 samples for internal validation; KarlssonFH_2013 supplied 96 samples as an external test cohort for the random forest baseline.")
    st.markdown("**Limitations:** The project does not establish a clinical diagnostic tool, clinical calibration, causal relationships between microbes and disease, or reliable generalisation across diverse populations. The displayed reference profiles are aggregates, not human samples. The original notebook's transformer was not externally validated on Karlsson in the saved evaluation results.")
    st.markdown("**Privacy and reproducibility:** Only trained weights, feature names, aggregated non-identifying examples and summary metrics are included in this deployment. Individual-level microbiome profiles, patient metadata and full clinical records are not bundled.")
    st.markdown("**Model:** 64-dimensional representations, 4 attention heads, 2 transformer layers, 5 rank bins, 289 selected species (as recorded in the supplied notebook).")
    st.caption("Project author: Aritra Sinha · Independent computational microbiome research")
