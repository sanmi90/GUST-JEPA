"""Config loader for the Session 31 loss kit (the fairness keystone).

``load_model_config`` deep-merges the ``configs/_kit.yaml`` defaults with a
per-model file and resolves it into a :class:`ResolvedKitConfig`, raising
:class:`KitConfigError` on any of the four fail-loud rules:

1. At most one of ``recon.on`` / ``pred.on`` is true. If neither is on, at least
   one supervision head must be on (the ``supervised_only`` control).
2. ``anti_collapse.on`` is mandatory when ``pred.on``. It is off for
   ``supervised_only`` (neither objective on). (The matched AEs hold it on in
   their own files; references such as Fukami legitimately set ``recon.on`` with
   anti-collapse off, so ``recon.on -> anti_collapse.on`` is NOT a universal
   rule and is not enforced here.)
3. ``representation_objective`` parameters (``recon.loss``, ``recon.field``,
   ``pred.horizon``, ``pred.loss``, ``pred.target``) and the pinned
   ``anti_collapse`` parameters (``lambda``, ``sigreg``, ``vicreg``,
   ``fallback``) come from ``_kit.yaml`` and may NOT be overridden per-model.
   Only ``on`` flags, the supervision ``on`` flags, ``anti_collapse.method`` and
   the passthrough ``model.*`` / ``name`` keys may vary per-model.
4. The ``training`` block is inherited from ``_kit.yaml``; a per-model file that
   sets any training key is rejected.

The kit deliberately does NOT support per-model loss-weight or field-loss
overrides: there must be no place outside ``_kit.yaml`` where a frozen parameter
is set (SESSION 31 Track A).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "KitConfigError",
    "ResolvedKitConfig",
    "load_model_config",
    "load_kit_yaml",
    "deep_merge",
]


class _KitYamlLoader(yaml.SafeLoader):
    """SafeLoader that does NOT coerce ``on``/``off``/``yes``/``no`` to booleans.

    The kit schema uses bare ``on:`` as a mapping key (e.g.
    ``recon: {on: false}``). Under the YAML 1.1 implicit resolver PyYAML reads
    ``on`` as the boolean ``True``, silently corrupting the key. This loader
    strips the broad boolean resolver and re-adds one that matches only
    ``true``/``false``, so ``on``/``off`` stay strings while ``true``/``false``
    values remain proper booleans.
    """


_KitYamlLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for (tag, rx) in resolvers if tag != "tag:yaml.org,2002:bool"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_KitYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_kit_yaml(path: str | Path) -> dict:
    """Load a kit YAML file with the ``on``-key-safe loader."""
    with open(path) as fh:
        return yaml.load(fh, Loader=_KitYamlLoader) or {}


class KitConfigError(ValueError):
    """Raised when a model config violates one of the four fail-loud kit rules."""


# Dotted leaf paths a per-model file is permitted to set. Anything else under
# representation_objective / anti_collapse / supervision / training is frozen.
_ALLOWED_OVERRIDE_LEAVES = frozenset(
    {
        "representation_objective.recon.on",
        "representation_objective.pred.on",
        "anti_collapse.on",
        "anti_collapse.method",
        "supervision.lift_head.on",
        "supervision.wake_head.on",
    }
)
# Whole subtrees a per-model file may carry freely (passthrough for Track B/C).
_ALLOWED_OVERRIDE_PREFIXES = ("model.",)
# Bare scalar keys allowed at the top level.
_ALLOWED_TOP_KEYS = frozenset({"name"})


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto a deep copy of ``base``.

    Nested dicts are merged key-by-key; any non-dict value in ``override``
    replaces the corresponding value in ``base``.
    """
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _leaf_paths(d: dict, prefix: str = "") -> list[str]:
    """Return dotted paths to every non-dict leaf in a nested mapping."""
    paths: list[str] = []
    for key, val in d.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            paths.extend(_leaf_paths(val, prefix=f"{path}."))
        else:
            paths.append(path)
    return paths


def _check_override_whitelist(override: dict, model_path: Path) -> None:
    """Rules 3 and 4: reject any per-model leaf outside the allowed set."""
    for leaf in _leaf_paths(override):
        if leaf in _ALLOWED_TOP_KEYS:
            continue
        if leaf in _ALLOWED_OVERRIDE_LEAVES:
            continue
        if any(leaf.startswith(p) for p in _ALLOWED_OVERRIDE_PREFIXES):
            continue
        if leaf.startswith("training."):
            raise KitConfigError(
                f"{model_path.name}: the `training` block is inherited from _kit.yaml "
                f"and may not be set per-model (rule 4); offending key '{leaf}'."
            )
        raise KitConfigError(
            f"{model_path.name}: '{leaf}' is a frozen kit parameter and may not be "
            f"overridden per-model (rule 3). A per-model file may only set "
            f"on-flags, anti_collapse.method, model.* passthrough keys, or name."
        )


@dataclass(frozen=True)
class ResolvedKitConfig:
    """A fully resolved, rule-validated model config.

    Attributes:
        name: Model name (from the per-model file, else the filename stem).
        representation_objective: Resolved ``{recon, pred}`` block.
        anti_collapse: Resolved anti-collapse block (``on``, ``method``,
            ``lambda``, ``sigreg``, ``vicreg``, ``fallback``).
        supervision: Resolved ``{lift_head, wake_head}`` block.
        training: Inherited optimiser / schedule / split / seed block.
        model: Passthrough encoder / latent flags (consumed by Track B/C).
        raw_override: The per-model override dict, kept for provenance.
    """

    name: str
    representation_objective: dict[str, Any]
    anti_collapse: dict[str, Any]
    supervision: dict[str, Any]
    training: dict[str, Any]
    model: dict[str, Any] = field(default_factory=dict)
    raw_override: dict[str, Any] = field(default_factory=dict)

    @property
    def objective(self) -> str:
        """``"recon"``, ``"pred"`` or ``"none"`` (the active representation axis)."""
        if self.representation_objective["recon"]["on"]:
            return "recon"
        if self.representation_objective["pred"]["on"]:
            return "pred"
        return "none"

    def active_terms(self) -> frozenset[str]:
        """The set of loss terms this model turns on.

        One of ``{"recon", "pred"}`` (or neither), plus any of
        ``{"anti_collapse", "lift", "wake"}``.
        """
        terms: set[str] = set()
        if self.representation_objective["recon"]["on"]:
            terms.add("recon")
        if self.representation_objective["pred"]["on"]:
            terms.add("pred")
        if self.anti_collapse["on"]:
            terms.add("anti_collapse")
        if self.supervision["lift_head"]["on"]:
            terms.add("lift")
        if self.supervision["wake_head"]["on"]:
            terms.add("wake")
        return frozenset(terms)


_ON_FLAG_PATHS = (
    "representation_objective.recon.on",
    "representation_objective.pred.on",
    "anti_collapse.on",
    "supervision.lift_head.on",
    "supervision.wake_head.on",
)


def _require_bool_flag(resolved: dict, dotted: str, model_path: Path) -> bool:
    """Return an on-flag, raising if it is not a real Python bool.

    The on-key-safe loader leaves bare ``on``/``off``/``yes``/``no`` as strings,
    and ``bool('off')`` is ``True``, so a string on-flag would silently turn a
    term on. This guard makes that fail loudly.
    """
    node: Any = resolved
    for key in dotted.split("."):
        node = node[key]
    if not isinstance(node, bool):
        raise KitConfigError(
            f"{model_path.name}: on-flag '{dotted}' must be a YAML boolean "
            f"(true/false), got {node!r} ({type(node).__name__}). Note that "
            f"'on'/'off'/'yes'/'no' are read as strings by the kit loader; "
            f"write true/false for on-flags."
        )
    return node


def _validate_rules(resolved: dict, model_path: Path) -> None:
    """Rules 1 and 2, checked on the resolved (merged) config."""
    recon_on, pred_on, ac_on, lift_on, wake_on = [
        _require_bool_flag(resolved, path, model_path) for path in _ON_FLAG_PATHS
    ]
    any_head = lift_on or wake_on

    # Rule 1
    if recon_on and pred_on:
        raise KitConfigError(
            f"{model_path.name}: at most one of recon.on / pred.on may be true "
            f"(rule 1); both are on."
        )
    if not recon_on and not pred_on and not any_head:
        raise KitConfigError(
            f"{model_path.name}: no representation objective is on, so at least one "
            f"supervision head must be on (rule 1, the supervised_only control)."
        )

    # Rule 2
    if pred_on and not ac_on:
        raise KitConfigError(
            f"{model_path.name}: anti_collapse.on is mandatory when pred.on (rule 2)."
        )
    if not recon_on and not pred_on and ac_on:
        raise KitConfigError(
            f"{model_path.name}: anti_collapse must be off for the supervised_only "
            f"control (no objective on) (rule 2)."
        )


def _default_kit_path(model_path: Path) -> Path:
    """Locate ``_kit.yaml`` as a sibling of the per-model file's parent directory.

    Per-model files live at ``configs/<group>/<model>.yaml`` so the kit defaults
    are at ``configs/_kit.yaml`` = ``model_path.parent.parent / "_kit.yaml"``.
    """
    return model_path.parent.parent / "_kit.yaml"


def load_model_config(path: str | Path, kit_path: str | Path | None = None) -> ResolvedKitConfig:
    """Deep-merge the kit defaults with a per-model file and validate the rules.

    Args:
        path: Path to the per-model YAML file.
        kit_path: Path to ``_kit.yaml``. Defaults to
            ``path.parent.parent / "_kit.yaml"``.

    Returns:
        A rule-validated :class:`ResolvedKitConfig`.

    Raises:
        KitConfigError: If any of the four fail-loud rules is violated.
        FileNotFoundError: If either YAML file is missing.
    """
    model_path = Path(path)
    kit_file = Path(kit_path) if kit_path is not None else _default_kit_path(model_path)
    if not kit_file.exists():
        raise FileNotFoundError(f"kit defaults not found at {kit_file}")
    if not model_path.exists():
        raise FileNotFoundError(f"model config not found at {model_path}")

    base = load_kit_yaml(kit_file)
    override = load_kit_yaml(model_path)

    # Rules 3 and 4 are about what the per-model file is allowed to SET, so they
    # are checked on the raw override before merging.
    _check_override_whitelist(override, model_path)

    resolved = deep_merge(base, override)
    _validate_rules(resolved, model_path)

    # Read the name from the per-model override only; falling back to the kit
    # default would always yield '_kit' (the merged base carries name: _kit).
    name = override.get("name", model_path.stem)
    return ResolvedKitConfig(
        name=str(name),
        representation_objective=resolved["representation_objective"],
        anti_collapse=resolved["anti_collapse"],
        supervision=resolved["supervision"],
        training=resolved.get("training", {}),
        model=resolved.get("model", {}),
        raw_override=override,
    )
