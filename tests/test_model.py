"""Tests for src/model.py's FiLM conditioning (step 6c). RUN THESE IN notebooks/06.

Needs segmentation_models_pytorch (in requirements-notebook.txt) but no
network access: every model here is built with ``encoder_weights=None``, so
nothing downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src import model as model_mod


def _settings(**overrides):
    settings = dict(model_mod.DEFAULTS)
    settings["film"] = dict(model_mod.DEFAULTS["film"])
    settings.update({k: v for k, v in overrides.items() if k != "film"})
    if "film" in overrides:
        settings["film"].update(overrides["film"])
    settings["encoder_weights"] = None   # no network access in a unit test
    return settings


def _config_with_film(tmp_path, film_value):
    """A full default.yaml with ONLY model.film overridden, in an isolated dir.

    Written into tmp_path, like tests/test_train.py's _config_with_exclusions,
    so the platform-overlay lookup beside the config file cannot pick up this
    repo's own configs/<platform>.yaml.
    """
    base = yaml.safe_load(
        (Path(model_mod.__file__).resolve().parents[1] / "configs" / "default.yaml")
        .read_text())
    base["model"] = dict(base["model"], film=film_value)
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(base))
    return path


# --------------------------------------------------------------------------
# config merge / validation
# --------------------------------------------------------------------------
def test_film_disabled_by_default():
    settings = model_mod.load_config()
    assert settings["film"] == {"enabled": False, "embed_dim": 16}


def test_film_config_merge_preserves_embed_dim_default(tmp_path):
    path = _config_with_film(tmp_path, {"enabled": True})
    settings = model_mod.load_config(path)
    assert settings["film"] == {"enabled": True, "embed_dim": 16}, (
        "setting only film.enabled must not silently lose film.embed_dim's "
        "default -- a bare dict replace would do exactly that")


def test_unknown_film_key_is_rejected(tmp_path):
    path = _config_with_film(tmp_path, {"enabled": True, "bogus": 1})
    with pytest.raises(model_mod.ModelError):
        model_mod.load_config(path)


def test_film_embed_dim_must_be_positive(tmp_path):
    path = _config_with_film(tmp_path, {"embed_dim": 0})
    with pytest.raises(model_mod.ModelError):
        model_mod.load_config(path)


# --------------------------------------------------------------------------
# build_model(film_vocabulary=...)
# --------------------------------------------------------------------------
def test_film_enabled_without_vocabulary_raises():
    settings = _settings(film={"enabled": True})
    with pytest.raises(model_mod.ModelError):
        model_mod.build_model(settings=settings)


def test_film_disabled_model_has_no_film_attributes():
    settings = _settings()
    model = model_mod.build_model(settings=settings)
    assert not hasattr(model, "film_vocabulary")
    assert not hasattr(model, "film_embedding")
    assert model.build_report["film"] == {"enabled": False}


def test_film_enabled_model_has_one_embedding_row_per_dataset():
    settings = _settings(film={"enabled": True, "embed_dim": 4})
    model = model_mod.build_model(settings=settings,
                                  film_vocabulary=["MetalDam", "Steel1"])
    assert model.film_vocabulary == {"MetalDam": 0, "Steel1": 1}
    assert model.film_embedding.weight.shape == (2, 4)
    assert model.build_report["film"]["enabled"] is True
    assert model.build_report["film"]["vocabulary"] == ["MetalDam", "Steel1"]


def test_film_vocabulary_is_deduplicated_and_sorted():
    settings = _settings(film={"enabled": True})
    model = model_mod.build_model(
        settings=settings, film_vocabulary=["Steel1", "MetalDam", "Steel1"])
    assert model.film_vocabulary == {"MetalDam": 0, "Steel1": 1}


# --------------------------------------------------------------------------
# _init_film()'s probe: batch size 1, encoder mode must not leak
# --------------------------------------------------------------------------
def test_film_init_succeeds_with_encoder_in_training_mode():
    """_init_film's probe forward pass used to crash here: build_model()
    never puts the encoder in eval mode before calling it, so the probe runs
    with the encoder in its default training mode and batch size 1. On
    resnet34, later stages collapse a 32x32 input to a 1x1 spatial map, and
    BatchNorm in training mode raises when a channel has only one value to
    compute a batch variance from ("Expected more than 1 value per channel
    when training, got input size ... 1, 1"). This is the real code path
    build_model() takes, not a contrived one.
    """
    settings = _settings(film={"enabled": True, "embed_dim": 4})
    model = model_mod.build_model(settings=settings,
                                  film_vocabulary=["MetalDam", "Steel1"])
    assert model.film_embedding.weight.shape == (2, 4)


def test_film_init_does_not_leak_eval_mode_onto_the_encoder():
    """The probe must flip the encoder to eval() only for its own forward
    pass and restore the mode it actually found -- build_model() leaves a
    freshly constructed encoder in training mode, so that is what must still
    be true once _init_film() returns.
    """
    settings = _settings(film={"enabled": True, "embed_dim": 4})
    model = model_mod.build_model(settings=settings,
                                  film_vocabulary=["MetalDam", "Steel1"])
    assert model.encoder.training is True


def test_film_init_restores_eval_mode_when_encoder_started_in_eval():
    """The restore must reproduce whichever mode the encoder was actually
    in, not unconditionally re-enable training mode.
    """
    settings = _settings(film={"enabled": True, "embed_dim": 4})
    model = model_mod.build_model(settings=settings,
                                  film_vocabulary=["MetalDam", "Steel1"])
    model.encoder.eval()
    model._init_film(["MetalDam", "Steel1"], 4, int(settings["in_channels"]))
    assert model.encoder.training is False


# --------------------------------------------------------------------------
# forward(): identity at init, embedding lookup, unknown-name fallback
# --------------------------------------------------------------------------
@pytest.fixture
def film_model():
    torch.manual_seed(0)
    settings = _settings(film={"enabled": True, "embed_dim": 4})
    model = model_mod.build_model(settings=settings,
                                  film_vocabulary=["MetalDam", "Steel1"])
    model.eval()
    return model


def test_film_is_the_identity_at_init(film_model):
    """Zero-initialised gamma=1,beta=0 must not move a single logit.

    The whole point of zero-init is that a freshly built FiLM model starts
    IDENTICAL to the unconditioned one -- conditioning is learned, not
    assumed. Same model, same weights: with no dataset info the FiLM branch
    is skipped entirely; with dataset_names given it runs but multiplies by
    1 and adds 0. The two forward passes must agree exactly.
    """
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        unconditioned = film_model(x)
        conditioned = film_model(x, dataset_names=["MetalDam", "Steel1"])
    assert torch.allclose(unconditioned, conditioned, atol=1e-6)


def test_embed_known_name_returns_its_own_row(film_model):
    with torch.no_grad():
        film_model.film_embedding.weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    got = film_model.embed(["MetalDam"])
    assert torch.allclose(got[0], film_model.film_embedding.weight[0])


def test_embed_unknown_name_returns_the_mean_training_embedding(film_model):
    """Validation is a MIXTURE that includes the held-out dataset, which by
    construction has no embedding of its own. This is what makes ordinary
    per-epoch validation (and therefore best.pt selection) work for a FiLM
    run without the caller having to know FiLM exists -- inference mode 1
    ("mean embedding"), applied automatically.
    """
    with torch.no_grad():
        film_model.film_embedding.weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    got = film_model.embed(["uhcs2"])   # not in the vocabulary
    expected = film_model.film_embedding.weight.mean(dim=0)
    assert torch.allclose(got[0], expected)


def test_embed_mixed_known_and_unknown_names(film_model):
    with torch.no_grad():
        film_model.film_embedding.weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    got = film_model.embed(["MetalDam", "uhcs2"])
    assert torch.allclose(got[0], film_model.film_embedding.weight[0])
    assert torch.allclose(got[1], film_model.film_embedding.weight.mean(dim=0))


def test_dataset_embedding_override_broadcasts_a_single_row(film_model):
    x = torch.randn(3, 1, 64, 64)
    row = torch.randn(1, 4)
    with torch.no_grad():
        out = film_model(x, dataset_embedding=row)
    assert out.shape[0] == 3


def test_dataset_embedding_batch_mismatch_raises(film_model):
    x = torch.randn(3, 1, 64, 64)
    row = torch.randn(2, 4)   # neither 1 nor 3
    with pytest.raises(model_mod.ModelError):
        film_model(x, dataset_embedding=row)


# --------------------------------------------------------------------------
# a FiLM model is still an ordinary model to every other src/model.py call
# --------------------------------------------------------------------------
def test_encoder_freeze_schedule_still_works_on_a_film_model(film_model):
    state = model_mod.apply_freeze_schedule(film_model, epoch=0,
                                            settings=_settings())
    assert state["frozen"] is True
    assert all(not p.requires_grad for p in film_model.encoder.parameters())

    state = model_mod.apply_freeze_schedule(film_model, epoch=10,
                                            settings=_settings())
    assert state["frozen"] is False
    assert all(p.requires_grad for p in film_model.encoder.parameters())


def test_summarize_and_describe_do_not_choke_on_a_film_model(film_model):
    params = model_mod.summarize(film_model)
    assert params["total"] > params["encoder_total"]
    text = model_mod.describe(film_model)
    assert "FiLM" in text
