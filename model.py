"""Original MiniBiomeGPT architecture and preprocessing reproduced from user's 2026 Colab notebook."""
import numpy as np
import torch
import torch.nn as nn


class MiniBiomeGPT(nn.Module):
    """2-layer transformer with species embeddings, abundance bins and CLS disease head."""
    def __init__(self, num_species):
        super().__init__()
        self.num_species = num_species
        self.species_embedding = nn.Embedding(num_species + 1, 64)
        self.abundance_embedding = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.species_norm = nn.LayerNorm(64)
        self.abundance_norm = nn.LayerNorm(64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.abundance_head = nn.Linear(64, 1)
        self.classifier = nn.Linear(64, 2)

    def forward(self, abundance_bins):
        batch_size, num_species = abundance_bins.shape
        if num_species != self.num_species:
            raise ValueError(f"Expected {self.num_species} species; got {num_species}")
        device = abundance_bins.device
        species_ids = torch.arange(num_species, device=device).unsqueeze(0).expand(batch_size, -1)
        cls_ids = torch.full((batch_size, 1), self.num_species, dtype=torch.long, device=device)
        species_ids = torch.cat([cls_ids, species_ids], dim=1)
        cls_bins = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        all_bins = torch.cat([cls_bins, abundance_bins], dim=1)
        species_vectors = self.species_embedding(species_ids)
        abundance_vectors = self.abundance_embedding(all_bins.float().unsqueeze(-1))
        tokens = self.species_norm(species_vectors) + self.abundance_norm(abundance_vectors)
        padding_mask = torch.cat([
            torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
            abundance_bins == 0
        ], dim=1)
        output = self.transformer(tokens, src_key_padding_mask=padding_mask)
        abundance_prediction = self.abundance_head(output[:, 1:, :]).squeeze(-1)
        disease_prediction = self.classifier(output[:, 0, :])
        return abundance_prediction, disease_prediction


def abundance_to_bins(sample, B=5):
    """Exactly match original Colab rank-within-sample abundance-to-bin conversion."""
    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim != 1 or not np.all(np.isfinite(sample)) or np.any(sample < 0):
        raise ValueError("Abundances must be a one-dimensional array of finite nonnegative values")
    bins = np.zeros(len(sample), dtype=np.int64)
    detected = np.where(sample > 0)[0]
    ranked = detected[np.argsort(-sample[detected])]
    groups = np.array_split(ranked, B)
    for group_number, group in enumerate(groups):
        bins[group] = B - group_number
    return bins


def load_checkpoint(path, expected_species):
    """Only load trusted exported checkpoints. No fallback to untrusted pickle."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state")
    num_species = int(checkpoint["num_species"])
    if num_species != len(expected_species):
        raise ValueError(f"Checkpoint requires {num_species} species, but feature list has {len(expected_species)}")
    saved_features = checkpoint.get("selected_species")
    if saved_features is not None and list(saved_features) != list(expected_species):
        raise ValueError("Checkpoint feature order differs from selected_species.json")
    model = MiniBiomeGPT(num_species)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def predict_probability(model, profile):
    bins = abundance_to_bins(profile)
    if not np.any(bins):
        raise ValueError("All species are zero. Supply a profile with at least one positive abundance.")
    x = torch.as_tensor(bins, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        _, logits = model(x)
        probability = torch.softmax(logits, dim=1)[0, 1].item()
    return float(probability), bins
