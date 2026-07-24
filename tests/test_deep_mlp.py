"""Shape/sanity checks for the DeepMLP classifier (Strategy 2) -- no GPU needed."""
import numpy as np
import torch

from aitext.classifiers.deep_mlp import DeepMLP, DeepMLPClassifier


def test_deep_mlp_forward_shape():
    model = DeepMLP(input_dim=5, hidden_dims=(8, 4), dropouts=(0.0, 0.0))
    x = torch.randn(3, 5)
    out = model(x)
    assert out.shape == (3, 2)


def test_deep_mlp_classifier_fit_predict_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    clf = DeepMLPClassifier(hidden_dims=(8, 4), dropouts=(0.0, 0.0), epochs=2, batch_size=8, seed=0)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (40, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
