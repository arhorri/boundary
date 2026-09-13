"""U-Net for boundary prediction: 1-channel input, raw logits out, no activation.

Three decisions are made here and every one of them is a correctness
constraint rather than a preference:

1. **The head returns logits.** No sigmoid, no softmax, nothing. The loss is
   ``BCEWithLogits``-based, which fuses the sigmoid into a numerically stable
   form; feeding it probabilities computes the sigmoid twice and silently
   destroys the gradient at saturation. ``activation=None`` is passed
   explicitly and the built graph is then WALKED to prove no activation module
   slipped in, because a wrong answer here does not raise -- it just trains
   badly.

2. **The first convolution is adapted, not replaced.** The encoder is
   pretrained on RGB ImageNet and the micrographs are single-channel. The
   three pretrained filters per output channel are SUMMED into one. Summing is
   what preserves the learned response: a filter that fires on an edge in R, G
   and B fires on that edge in gray with the same magnitude. Taking one
   channel throws away two thirds of the evidence, and reinitialising throws
   away the pretraining entirely -- which is the only reason to use an
   ImageNet encoder on grayscale SEM images in the first place.

   ``segmentation_models_pytorch`` ALREADY does the sum for ``in_channels=1``
   (``smp.encoders._utils.patch_first_conv``: ``weight.sum(1, keepdim=True)``
   in that branch). This module does not take that on trust across versions.
   It builds a reference 3-channel encoder, sums its first-conv weights, and
   compares. If smp's result matches, nothing is overridden and the report
   says so; if it does not, the summed weights are written in and the report
   says THAT. Either way ``model.build_report`` states what actually happened
   rather than what was assumed.

3. **The encoder can be frozen for the first N epochs.** The decoder starts
   random. Its first gradients are noise, and letting that noise into a
   pretrained encoder in epoch 1 is how the pretraining gets erased before it
   is ever used. Freezing also has to stop BatchNorm from updating its running
   statistics -- a "frozen" encoder whose BN layers are still in train mode is
   not frozen, it is drifting, and nothing about it raises.

Nothing here trains, loads a checkpoint or touches data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


class ModelError(RuntimeError):
    """Raised when a model cannot be built as specified. Never fails silently."""


try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - hosts ship torch
    raise ModelError(
        "torch is not importable. Colab and Kaggle ship it preinstalled; this "
        "module is meant to run on a host, never on a local machine."
    ) from exc


# --------------------------------------------------------------------------
# defaults -- overridable from configs/default.yaml under ``model:``
# --------------------------------------------------------------------------
DEFAULTS = {
    "arch": "unet",
    "encoder": "resnet34",
    "encoder_weights": "imagenet",   # null = random init, no adaptation needed
    "in_channels": 1,                # micrographs are single-channel
    "classes": 1,                    # boundary / not-boundary, one logit map
    # Epochs the encoder stays frozen at the start of training. The decoder is
    # random; its opening gradients are noise. 3 is a starting point for step 6
    # to confirm, not a measured value.
    "freeze_encoder_epochs": 3,
    # Per-dataset FiLM conditioning (step 6c), off by default. See
    # build_model(film_vocabulary=...) and _FiLMUnetMixin below. The embedding
    # table is sized by the CALLER's vocabulary (Trainer resolves it from the
    # fold's train_datasets), not by anything in this dict.
    "film": {"enabled": False, "embed_dim": 16},
}

#: Architectures this project will build. Anything else is a typo, not a feature.
ARCHITECTURES = ("unet", "unetplusplus")

#: Modules that would mean the head is not returning logits.
FORBIDDEN_ACTIVATIONS = ("Sigmoid", "Softmax", "Softmax2d", "LogSigmoid",
                         "LogSoftmax", "Tanh")


def load_config(config_path: Optional[Path] = None) -> dict:
    """Merge ``model:`` from configs/default.yaml over DEFAULTS."""
    from src import paths as paths_mod

    cfg = paths_mod.load_config(config_path)
    settings = dict(DEFAULTS)
    section = cfg.get("model") or {}
    if not isinstance(section, dict):
        raise ModelError(
            f"configs/default.yaml: model must be a mapping, got "
            f"{type(section).__name__}")
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise ModelError(
            f"configs/default.yaml: unknown model keys {sorted(unknown)}; "
            f"known keys are {sorted(DEFAULTS)}")
    for key, value in section.items():
        # encoder_weights: null is a real choice (random init), so it is only
        # honoured when the key is present at all.
        if key == "film" and isinstance(value, dict):
            # A nested dict is MERGED, not replaced -- a config that sets only
            # film.enabled must not silently lose film.embed_dim's default.
            unknown_film = set(value) - set(DEFAULTS["film"])
            if unknown_film:
                raise ModelError(
                    f"configs/default.yaml: unknown model.film keys "
                    f"{sorted(unknown_film)}; known keys are "
                    f"{sorted(DEFAULTS['film'])}")
            merged = dict(DEFAULTS["film"])
            merged.update(value)
            settings[key] = merged
        elif value is not None or key == "encoder_weights":
            settings[key] = value
    settings["film"] = dict(settings["film"])
    settings["film"]["enabled"] = bool(settings["film"]["enabled"])
    settings["film"]["embed_dim"] = int(settings["film"]["embed_dim"])
    if settings["film"]["embed_dim"] < 1:
        raise ModelError(
            f"model.film.embed_dim must be >= 1, got "
            f"{settings['film']['embed_dim']}")
    return settings


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def _smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ModelError(
            "segmentation_models_pytorch is not importable. It is listed in "
            "requirements-notebook.txt and installed by the bootstrap cell; "
            "re-run cell 1."
        ) from exc
    return smp


def _architecture(settings: dict):
    """The smp class named by ``model.arch``. Unknown names fail loudly."""
    smp = _smp()
    name = str(settings["arch"]).strip().lower()
    if name not in ARCHITECTURES:
        raise ModelError(
            f"model.arch is {settings['arch']!r}; this project builds "
            f"{list(ARCHITECTURES)}. Add the class here deliberately rather "
            "than passing an arbitrary name through to smp.")
    return {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}[name]


def first_conv(module: "nn.Module") -> "nn.Conv2d":
    """The first Conv2d in registration order -- the one that sees the input."""
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    raise ModelError(
        f"no Conv2d found in {type(module).__name__}; the encoder cannot be "
        "the one this project thinks it is.")


def assert_returns_logits(model: "nn.Module") -> list:
    """Walk the built graph and prove no activation was attached to the head.

    ``activation=None`` is passed at construction, but a default that changes
    in a later smp release would turn every logit into a probability and
    nothing would raise: the loss would just stop working properly. So the
    claim is checked against the object, not the argument.
    """
    found = [f"{name or '<root>'}: {type(m).__name__}"
             for name, m in model.named_modules()
             if type(m).__name__ in FORBIDDEN_ACTIVATIONS]
    if found:
        raise ModelError(
            "the model contains activation modules that would stop it "
            f"returning logits: {found}. The loss applies its own sigmoid; a "
            "second one saturates the gradient. Build with activation=None.")
    return found


def _pretrained_rgb_first_conv(settings: dict) -> "torch.Tensor":
    """First-conv weights of an UNMODIFIED 3-channel encoder, for comparison.

    Two ways of getting it, because this is the reference the whole grayscale
    adaptation is judged against and losing it silently is not acceptable:
    ``smp.encoders.get_encoder`` first (cheap, just the encoder), then a full
    3-channel model of the configured architecture (certain to work, since
    that is the call the real model makes). If both fail the caller is told to
    disable verification explicitly rather than being handed an unverified
    model that claims to be verified.
    """
    smp = _smp()
    name = settings["encoder"]
    weights = settings["encoder_weights"]
    errors = []
    try:
        ref = smp.encoders.get_encoder(name, in_channels=3, depth=5,
                                       weights=weights)
    except Exception as exc:                      # noqa: BLE001 - reported below
        errors.append(f"smp.encoders.get_encoder: {type(exc).__name__}: {exc}")
        try:
            ref = _architecture(settings)(
                encoder_name=name, encoder_weights=weights, in_channels=3,
                classes=int(settings["classes"]), activation=None).encoder
        except Exception as exc2:                 # noqa: BLE001 - reported below
            errors.append(f"3-channel {settings['arch']}: "
                          f"{type(exc2).__name__}: {exc2}")
            raise ModelError(
                "cannot build a reference 3-channel encoder, so the claim "
                "that the pretrained RGB filters were summed into one channel "
                "cannot be verified:\n  " + "\n  ".join(errors)
                + "\nPass verify_first_conv=False to build anyway -- and then "
                "do not claim in the report that the adaptation was checked."
            ) from exc2
    return first_conv(ref).weight.detach().clone()


def adapt_first_conv(model: "nn.Module", settings: dict,
                     verify: bool = True) -> dict:
    """Make sure the first conv holds the SUM of the pretrained RGB filters.

    Returns a report describing what was found and whether anything had to be
    written. ``strategy`` is one of:

    ``random``      no pretrained weights were requested; nothing to preserve.
    ``rgb-native``  ``in_channels`` is 3; the conv is untouched.
    ``sum``         the conv already held the summed filters (what smp does).
    ``sum-forced``  it did not, and this function wrote them in.
    ``unverified``  verification was switched off by the caller.
    """
    conv = first_conv(model.encoder)
    in_channels = int(settings["in_channels"])
    report = {
        "encoder": settings["encoder"],
        "encoder_weights": settings["encoder_weights"],
        "in_channels": in_channels,
        "first_conv": f"{tuple(conv.weight.shape)} groups={conv.groups}",
        "overridden": False,
        "verified": False,
        "smp_did": None,
        "strategy": None,
        "note": "",
    }

    if conv.weight.shape[1] != in_channels:
        raise ModelError(
            f"the first conv takes {conv.weight.shape[1]} channels but "
            f"model.in_channels is {in_channels}. smp did not adapt the "
            "encoder as expected; do not train on this.")

    if settings["encoder_weights"] in (None, "none", ""):
        report["strategy"] = "random"
        report["note"] = ("no pretrained weights requested, so there is no "
                          "RGB filter to preserve")
        return report
    if in_channels == 3:
        report["strategy"] = "rgb-native"
        report["note"] = "input is already 3-channel; no adaptation applies"
        return report
    if in_channels != 1:
        raise ModelError(
            f"model.in_channels is {in_channels}. This project's adaptation is "
            "defined for 1 (sum the RGB filters) and 3 (leave them alone). "
            "Anything else needs a decision, not a default.")
    if conv.groups != 1:
        raise ModelError(
            f"the first conv is grouped (groups={conv.groups}); summing input "
            "channels is not defined for it. Choose an encoder whose stem is a "
            "plain convolution.")
    if not verify:
        report["strategy"] = "unverified"
        report["note"] = ("verification disabled by the caller; whatever smp "
                          "did was kept")
        return report

    rgb = _pretrained_rgb_first_conv(settings)
    if rgb.shape[1] != 3:
        raise ModelError(
            f"the reference encoder's first conv takes {rgb.shape[1]} input "
            "channels, not 3; the RGB assumption behind this adaptation does "
            "not hold for it.")
    expected = rgb.sum(dim=1, keepdim=True)
    got = conv.weight.detach().to(expected.dtype)

    # Which of the three plausible things smp actually did -- reported rather
    # than assumed, because they are indistinguishable from the outside.
    def _close(a, b):
        return bool(torch.allclose(a, b, rtol=0.0, atol=1e-6))

    if _close(got, expected):
        report["smp_did"] = "sum"
    elif _close(got, rgb[:, :1]):
        report["smp_did"] = "first-channel slice"
    elif _close(got, rgb.mean(dim=1, keepdim=True)):
        report["smp_did"] = "mean"
    else:
        report["smp_did"] = "something else (or reinitialised)"

    report["verified"] = True
    report["max_abs_diff_from_sum"] = float((got - expected).abs().max())
    if report["smp_did"] == "sum":
        report["strategy"] = "sum"
        report["note"] = ("smp already summed the pretrained RGB filters; "
                          "nothing overridden")
    else:
        with torch.no_grad():
            conv.weight.copy_(expected.to(conv.weight.dtype))
        report["overridden"] = True
        report["strategy"] = "sum-forced"
        report["note"] = (f"smp produced a {report['smp_did']}; the summed RGB "
                          "filters were written in instead")
    report["response_ratio"] = float(expected.abs().sum() / rgb.abs().sum())
    return report


# --------------------------------------------------------------------------
# conditioning: per-dataset FiLM (step 6c)
# --------------------------------------------------------------------------
class _FiLMUnetMixin:
    """Feature-wise linear modulation of the encoder's feature maps.

    Mixed into a fresh class (``type(f"FiLM{cls.__name__}", (_FiLMUnetMixin,
    cls), {})``, see :func:`_film_class`) rather than wrapping a built
    instance, so ``model.encoder`` / ``model.decoder`` / ``model.segmentation_head``
    stay top-level attributes -- every existing freeze-schedule, optimizer
    param-grouping and checkpoint-checksum call that reaches into those needs
    no change.

    Every FiLM linear layer is ZERO-initialised (gamma = 1 + 0, beta = 0), so a
    freshly built FiLM model is the IDENTITY of the unconditioned one until it
    trains: conditioning is learned, not assumed, and a from-scratch comparison
    against the unconditioned baseline starts from the same point.
    """

    def _init_film(self, vocabulary: Sequence[str], embed_dim: int,
                   in_channels: int) -> None:
        if not vocabulary:
            raise ModelError(
                "film_vocabulary is empty; FiLM needs at least one training "
                "dataset to build an embedding table for.")
        channels = tuple(int(c) for c in self.encoder.out_channels)
        # Verified against a real forward pass, not assumed: an smp encoder
        # whose out_channels disagreed with what encoder(x) actually returns
        # would silently modulate the wrong tensor shapes.
        with torch.no_grad():
            probe = torch.zeros(1, int(in_channels), 32, 32)
            try:
                features = self.encoder(probe)
            except Exception as exc:                       # noqa: BLE001
                raise ModelError(
                    "could not probe the encoder to size FiLM's per-stage "
                    f"layers: {type(exc).__name__}: {exc}") from exc
        if len(features) != len(channels):
            raise ModelError(
                f"encoder.out_channels has {len(channels)} entries but a "
                f"probe forward pass returned {len(features)} feature maps; "
                "FiLM's per-stage layers would not line up with the real "
                "features. Re-check this against the installed "
                "segmentation_models_pytorch version on this host.")
        for i, (feature, c) in enumerate(zip(features, channels)):
            if feature.shape[1] != c:
                raise ModelError(
                    f"encoder.out_channels[{i}]={c} but the probe feature map "
                    f"has {feature.shape[1]} channels; FiLM cannot be sized "
                    "from encoder.out_channels for this encoder.")

        self.film_vocabulary = {name: i for i, name in enumerate(vocabulary)}
        self.film_embedding = nn.Embedding(len(vocabulary), embed_dim)
        self.film_layers = nn.ModuleList(
            nn.Linear(embed_dim, 2 * c) for c in channels)
        for layer in self.film_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def embed(self, dataset_names: Sequence[str]) -> "torch.Tensor":
        """One embedding row per name.

        A name outside the vocabulary is NOT an error here: validation is a
        MIXTURE that includes the held-out dataset, which by construction has
        no embedding of its own (it was never trained on). Such a name gets
        the MEAN of the training vocabulary's embeddings -- this is
        inference mode 1 ("mean embedding"), applied automatically so
        ordinary per-epoch validation (and therefore best.pt selection) works
        without the caller having to know FiLM exists. The explicit
        ``dataset_embedding=`` override on :meth:`forward` is how modes 2
        ("each training dataset in turn") and 3 ("closest") are scored
        instead, post-hoc, in :func:`src.train.evaluate_film_inference_modes`.
        """
        device = self.film_embedding.weight.device
        mean_embedding = self.film_embedding.weight.mean(dim=0)
        rows = [self.film_embedding.weight[self.film_vocabulary[name]]
               if name in self.film_vocabulary else mean_embedding
               for name in dataset_names]
        return torch.stack(rows, dim=0).to(device)

    @staticmethod
    def _apply_film(feature: "torch.Tensor", layer: "nn.Linear",
                    embedding: "torch.Tensor") -> "torch.Tensor":
        gamma_beta = layer(embedding)
        c = feature.shape[1]
        gamma = 1.0 + gamma_beta[:, :c].reshape(-1, c, 1, 1)
        beta = gamma_beta[:, c:].reshape(-1, c, 1, 1)
        return gamma * feature + beta

    def forward(self, x: "torch.Tensor",
               dataset_names: Optional[Sequence[str]] = None,
               dataset_embedding: Optional["torch.Tensor"] = None):
        if dataset_embedding is None and dataset_names is not None:
            dataset_embedding = self.embed(dataset_names)
        features = self.encoder(x)
        if dataset_embedding is not None:
            batch = x.shape[0]
            if dataset_embedding.shape[0] == 1 and batch != 1:
                dataset_embedding = dataset_embedding.expand(batch, -1)
            elif dataset_embedding.shape[0] != batch:
                raise ModelError(
                    f"dataset_embedding has batch size "
                    f"{dataset_embedding.shape[0]} but x has batch size "
                    f"{batch}; pass one embedding row per input image, or a "
                    "single row to broadcast to the whole batch.")
            features = [self._apply_film(f, layer, dataset_embedding)
                       for f, layer in zip(features, self.film_layers)]
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)


def _film_class(base_cls):
    """A subclass of ``base_cls`` with :class:`_FiLMUnetMixin` mixed in.

    A fresh class per call rather than a module-level constant: ``base_cls``
    is chosen at build time by ``model.arch`` (Unet or UnetPlusPlus), and the
    mixin must come first in the MRO so its ``forward`` overrides the base
    class's.
    """
    return type(f"FiLM{base_cls.__name__}", (_FiLMUnetMixin, base_cls), {})


def build_model(settings: Optional[dict] = None,
                config_path: Optional[Path] = None,
                verify_first_conv: bool = True,
                film_vocabulary: Optional[Sequence[str]] = None) -> "nn.Module":
    """Build the segmentation model described by ``model:`` in the config.

    Returns the smp model itself -- not a wrapper -- so checkpoints stay
    interoperable with smp and with anything else that reads them. What was
    done to it is attached as ``model.build_report``, a plain dict (not a
    buffer or a submodule, so it never reaches ``state_dict``).

    ``film_vocabulary`` is required when ``model.film.enabled`` is true: the
    embedding table's size is a property of the FOLD being trained (its
    resolved training datasets after exclusion), which this module has no way
    to know on its own -- only ``Trainer`` does. Passing it explicitly keeps
    that fact visible at the call site rather than reached for implicitly.
    """
    settings = settings or load_config(config_path)
    in_channels = int(settings["in_channels"])
    classes = int(settings["classes"])
    if in_channels < 1 or classes < 1:
        raise ModelError(
            f"model.in_channels={in_channels}, model.classes={classes}; both "
            "must be at least 1.")
    if classes != 1:
        raise ModelError(
            f"model.classes is {classes}. This step predicts ONE boundary "
            "logit map and the loss is binary; a multi-class head would need "
            "a different loss, not just a different number here.")

    cls = _architecture(settings)
    film_settings = settings.get("film") or DEFAULTS["film"]
    film_enabled = bool(film_settings.get("enabled", False))
    if film_enabled:
        if not film_vocabulary:
            raise ModelError(
                "model.film.enabled is true but build_model() was not given "
                "film_vocabulary; the embedding table size depends on the "
                "fold's resolved training datasets, which only the caller "
                "(Trainer) knows.")
        cls = _film_class(cls)

    weights = settings["encoder_weights"]
    model = cls(
        encoder_name=settings["encoder"],
        encoder_weights=(None if weights in ("none", "") else weights),
        in_channels=in_channels,
        classes=classes,
        activation=None,   # logits. Checked below, not trusted.
    )
    vocabulary = sorted(set(film_vocabulary)) if film_enabled else None
    if film_enabled:
        model._init_film(vocabulary, int(film_settings["embed_dim"]), in_channels)
    assert_returns_logits(model)
    report = adapt_first_conv(model, settings, verify=verify_first_conv)
    report["arch"] = settings["arch"]
    report["classes"] = classes
    report["freeze_encoder_epochs"] = int(settings["freeze_encoder_epochs"])
    report["film"] = ({"enabled": True, "vocabulary": vocabulary,
                       "embed_dim": int(film_settings["embed_dim"])}
                      if film_enabled else {"enabled": False})
    report["parameters"] = summarize(model)
    model.build_report = report
    return model


# --------------------------------------------------------------------------
# parameter accounting
# --------------------------------------------------------------------------
def _count(module) -> tuple:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def summarize(model: "nn.Module") -> dict:
    """Parameter counts, split encoder / decoder+head, total and trainable."""
    total, trainable = _count(model)
    enc_total, enc_trainable = _count(model.encoder)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "encoder_total": enc_total,
        "encoder_trainable": enc_trainable,
        "decoder_and_head_total": total - enc_total,
    }


# --------------------------------------------------------------------------
# encoder freezing
# --------------------------------------------------------------------------
def _encoder_norm_layers(model: "nn.Module") -> list:
    return [m for m in model.encoder.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def set_encoder_trainable(model: "nn.Module", trainable: bool,
                          bn_eval: Optional[bool] = None) -> dict:
    """Freeze or unfreeze the encoder, INCLUDING its BatchNorm statistics.

    ``requires_grad = False`` stops the weights being updated by the optimizer.
    It does NOT stop BatchNorm updating ``running_mean`` and ``running_var``,
    which happens in the forward pass whenever the module is in train mode. An
    encoder frozen that way still drifts every batch, and the drift is
    invisible: no gradient, no error, just a slowly changing feature extractor
    underneath a decoder that is trying to fit it.

    So the norm layers are put in eval mode too, by default whenever the
    encoder is frozen.

    **This must be re-applied after every ``model.train()`` call**, because
    ``train()`` walks the whole tree and puts those layers back into train
    mode. :func:`apply_freeze_schedule` is the per-epoch call that does it.
    """
    bn_eval = (not trainable) if bn_eval is None else bool(bn_eval)
    for param in model.encoder.parameters():
        param.requires_grad = bool(trainable)
    norms = _encoder_norm_layers(model)
    for module in norms:
        module.train(not bn_eval)
    return {
        "trainable": bool(trainable),
        "bn_eval": bn_eval,
        "norm_layers": len(norms),
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
    }


def freeze_encoder(model: "nn.Module") -> dict:
    return set_encoder_trainable(model, False)


def unfreeze_encoder(model: "nn.Module") -> dict:
    return set_encoder_trainable(model, True)


def apply_freeze_schedule(model: "nn.Module", epoch: int,
                          settings: Optional[dict] = None) -> dict:
    """Freeze the encoder for the first N epochs, unfreeze from epoch N on.

    Call it once per epoch, AFTER ``model.train()``, before the first batch.
    Idempotent: ``changed`` says whether this call actually flipped anything,
    so a training loop can log the transition once instead of every epoch.
    """
    settings = settings or load_config()
    n = int(settings["freeze_encoder_epochs"])
    if epoch < 0:
        raise ModelError(f"epoch must be >= 0, got {epoch}")
    should_freeze = epoch < n
    was_trainable = any(p.requires_grad for p in model.encoder.parameters())
    state = set_encoder_trainable(model, not should_freeze)
    state.update({
        "epoch": int(epoch),
        "freeze_encoder_epochs": n,
        "frozen": should_freeze,
        "changed": was_trainable == should_freeze,
    })
    return state


def encoder_state(model: "nn.Module") -> dict:
    """What is actually frozen right now -- measured, not remembered."""
    params = list(model.encoder.parameters())
    norms = _encoder_norm_layers(model)
    return {
        "encoder_trainable_params": sum(p.numel() for p in params if p.requires_grad),
        "encoder_frozen_params": sum(p.numel() for p in params if not p.requires_grad),
        "norm_layers": len(norms),
        "norm_layers_in_eval": sum(1 for m in norms if not m.training),
        "fully_frozen": all(not p.requires_grad for p in params)
                        and all(not m.training for m in norms),
    }


def describe(model: "nn.Module") -> str:
    """One printable block: what was built, how it was adapted, what is frozen."""
    report = getattr(model, "build_report", {}) or {}
    params = summarize(model)
    state = encoder_state(model)
    lines = [
        f"arch            : {report.get('arch', '?')}",
        f"encoder         : {report.get('encoder', '?')} "
        f"(weights={report.get('encoder_weights', '?')})",
        f"in_channels     : {report.get('in_channels', '?')}   "
        f"classes: {report.get('classes', '?')}   activation: none (logits)",
        f"first conv      : {report.get('first_conv', '?')}",
        f"1-channel adapt : {report.get('strategy', '?')} -- {report.get('note', '')}",
        f"  smp produced  : {report.get('smp_did')}"
        f"   verified: {report.get('verified')}"
        f"   overridden: {report.get('overridden')}",
        f"parameters      : {params['total']:,} total, "
        f"{params['trainable']:,} trainable, {params['frozen']:,} frozen",
        f"  encoder       : {params['encoder_total']:,} "
        f"({params['encoder_trainable']:,} trainable)",
        f"  decoder+head  : {params['decoder_and_head_total']:,}",
        f"encoder frozen  : {state['fully_frozen']} "
        f"({state['norm_layers_in_eval']}/{state['norm_layers']} norm layers in eval)",
    ]
    film = report.get("film") or {"enabled": False}
    if film.get("enabled"):
        lines.append(
            f"FiLM            : enabled, embed_dim={film.get('embed_dim')}, "
            f"vocabulary={film.get('vocabulary')}")
    else:
        lines.append("FiLM            : disabled")
    return "\n".join(lines)
