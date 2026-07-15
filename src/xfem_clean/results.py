"""Stable public result boundary for single- and multi-crack analyses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any
import warnings as python_warnings

import numpy as np


RESULT_SCHEMA_VERSION = 1
HISTORY_COLUMNS = (
    "step", "u", "P", "M", "kappa", "R", "crack_tip_x", "crack_tip_y",
    "angle_deg", "crack_active", "W_plastic", "W_damage_t", "W_damage_c",
    "W_cohesive", "W_total",
)


def normalize_history_rows(history: Any) -> tuple[dict[str, Any], ...]:
    """Normalize supported solver histories without copying large field arrays."""
    if history is None:
        return ()
    if isinstance(history, np.ndarray):
        rows = history.tolist()
    else:
        rows = list(history)
    if not rows:
        return ()

    normalized = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            item = dict(row)
            item.setdefault("step", index)
        else:
            values = np.asarray(row).reshape(-1).tolist()
            item = {name: values[i] for i, name in enumerate(HISTORY_COLUMNS[:len(values)])}
            item.setdefault("step", index)
        normalized.append(item)
    return tuple(normalized)


@dataclass(slots=True)
class AnalysisResult(Mapping[str, Any]):
    """Versioned common result with an explicit deprecated legacy mapping view.

    Public step units are SI: ``u`` [m], ``P`` [N], ``M`` [N m], and work or
    energy terms [J]. Field arrays are held by reference to avoid large copies.
    """

    steps: tuple[dict[str, Any], ...]
    fields: Mapping[str, Any]
    energies_or_work: Mapping[str, Any]
    cracks: tuple[Any, ...]
    reinforcement: Mapping[str, Any]
    solver_meta: Mapping[str, Any]
    warnings: tuple[Mapping[str, Any], ...] = ()
    schema_version: int = RESULT_SCHEMA_VERSION
    _legacy: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_solver_bundle(
        cls,
        bundle: Mapping[str, Any],
        *,
        engine: str,
        material: str,
        bond_layer_count: int,
        use_numba: bool,
        compat_mode: bool,
        structured_warnings: tuple[Mapping[str, Any], ...] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> "AnalysisResult":
        steps = normalize_history_rows(bundle.get("history"))
        cracks = tuple(bundle.get("cracks") or ())
        if not cracks and bundle.get("crack") is not None:
            cracks = (bundle["crack"],)

        final = steps[-1] if steps else {}
        energy = {
            "plastic_dissipation_J": final.get("W_plastic"),
            "tension_damage_dissipation_J": final.get("W_damage_t"),
            "compression_damage_dissipation_J": final.get("W_damage_c"),
            "cohesive_fracture_energy_J": final.get("W_cohesive"),
            "total_irreversible_energy_J": final.get("W_total"),
            "interface_work_J": final.get("interface_work"),
            "bond_work_J": final.get("bond_work"),
        }
        energy = {key: value for key, value in energy.items() if value is not None}

        fields = {
            "nodes_m": bundle.get("nodes"),
            "elements": bundle.get("elems"),
            "displacement_m": bundle.get("u"),
            "material_points": bundle.get("mp_states", bundle.get("bulk_states")),
            "cohesive_states": bundle.get("coh_states"),
        }
        reinforcement = {
            "bond_states": bundle.get("bond_states"),
            "segments": bundle.get("rebar_segs"),
            "bond_layer_count": int(bond_layer_count),
        }
        meta = {
            "engine": engine,
            "crack_path": "multi" if engine == "multi" else "single",
            "material": material,
            "bond_layer_count": int(bond_layer_count),
            "use_numba": bool(use_numba),
            "compat_mode": bool(compat_mode),
            "converged": bool(steps),
            "step_count": len(steps),
            "provenance": dict(provenance or {}),
        }
        return cls(
            steps=steps,
            fields=fields,
            energies_or_work=energy,
            cracks=cracks,
            reinforcement=reinforcement,
            solver_meta=meta,
            warnings=structured_warnings,
            _legacy=dict(bundle),
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the historical solver bundle for maintained old consumers."""
        return dict(self._legacy)

    @property
    def legacy_view(self) -> Mapping[str, Any]:
        python_warnings.warn(
            "AnalysisResult.legacy_view is deprecated; consume the canonical fields instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._legacy

    def __getitem__(self, key: str) -> Any:
        canonical = {
            "steps": self.steps,
            "fields": self.fields,
            "energies_or_work": self.energies_or_work,
            "cracks": self.cracks,
            "reinforcement": self.reinforcement,
            "solver_meta": self.solver_meta,
            "warnings": self.warnings,
            "schema_version": self.schema_version,
        }
        if key in canonical:
            return canonical[key]
        return self._legacy[key]

    def __iter__(self) -> Iterator[str]:
        seen = set()
        for key in (
            "schema_version", "steps", "fields", "energies_or_work", "cracks",
            "reinforcement", "solver_meta", "warnings",
        ):
            seen.add(key)
            yield key
        for key in self._legacy:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(set(self._legacy) | {
            "schema_version", "steps", "fields", "energies_or_work", "cracks",
            "reinforcement", "solver_meta", "warnings",
        })
