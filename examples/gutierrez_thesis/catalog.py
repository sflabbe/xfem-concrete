"""Canonical registry and compatibility declarations for thesis examples.

The Python factories are the only maintained definitions.  This module keeps
identity, aliases, execution status, and engine capability separate from the
physical values stored by each factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from examples.gutierrez_thesis.case_config import CaseConfig
from examples.gutierrez_thesis.cases.case_01_pullout_lettow import create_case_01
from examples.gutierrez_thesis.cases.case_02_sspot_frp import create_case_02
from examples.gutierrez_thesis.cases.case_03_tensile_stn12 import create_case_03
from examples.gutierrez_thesis.cases.case_04_beam_3pb_t5a1 import create_case_04
from examples.gutierrez_thesis.cases.case_04a_beam_3pb_t5a1_bosco import create_case_04a
from examples.gutierrez_thesis.cases.case_04b_beam_3pb_t6a1_bosco import create_case_04b
from examples.gutierrez_thesis.cases.case_05_wall_c1_cyclic import create_case_05
from examples.gutierrez_thesis.cases.case_06_fibre_tensile import create_case_06
from examples.gutierrez_thesis.cases.case_07_beam_4pb_jason_4pbt import create_case_07
from examples.gutierrez_thesis.cases.case_08_beam_3pb_vvbs3_cfrp import create_case_08
from examples.gutierrez_thesis.cases.case_09_beam_4pb_fibres_sorelli import create_case_09
from examples.gutierrez_thesis.cases.case_10_wall_c2_cyclic import create_case_10
from examples.gutierrez_thesis.cases.case_11_balcony_cantilever_sls import create_case_11


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    factory: Callable[[], CaseConfig]
    aliases: tuple[str, ...]
    status: str
    validation_source: str
    note: str

    @property
    def factory_source(self) -> str:
        return f"factory:{self.factory.__module__}.{self.factory.__name__}"


CASE_DEFINITIONS = (
    CaseDefinition(
        "01_pullout_lettow", create_case_01, ("pullout", "lettow"),
        "operational-smoke", "Lettow 2006; factory comments",
        "Elastic multicrack path with one embedded bond layer.",
    ),
    CaseDefinition(
        "02_sspot_frp", create_case_02, ("sspot", "frp"),
        "operational-smoke", "factory comments; archived SSPOT notes",
        "Elastic multicrack path with one FRP bond layer.",
    ),
    CaseDefinition(
        "03_tensile_stn12", create_case_03, ("stn12", "tensile"),
        "experimental-benchmark", "Gutiérrez thesis mapping; factory comments",
        "CDP-lite multicrack; requires the partial Numba backend.",
    ),
    CaseDefinition(
        "04_beam_3pb_t5a1", create_case_04, ("beam", "3pb"),
        "legacy-active-unsupported", "legacy factory only",
        "Retained alias target; cdp_full is not faithfully supported by multicrack.",
    ),
    CaseDefinition(
        "04a_beam_3pb_t5a1_bosco", create_case_04a, ("t5a1", "bosco_t5a1"),
        "ambiguous-unsupported", "factory + archived YAML; conflicting local documentation",
        "Factory/YAML describe 4000x400 mm and 4Ø12+2Ø10; README/reference placeholders describe 1500x250 mm and 2Ø16. cdp_full is also unsupported by multicrack.",
    ),
    CaseDefinition(
        "04b_beam_3pb_t6a1_bosco", create_case_04b, ("t6a1", "bosco_t6a1"),
        "experimental-unsupported", "factory comments",
        "Derived BOSCO benchmark; cdp_full is not faithfully supported by multicrack.",
    ),
    CaseDefinition(
        "05_wall_c1_cyclic", create_case_05, ("wall", "c1", "cyclic"),
        "experimental-unsupported", "factory comments",
        "Cyclic wall definition; cdp_full is not faithfully supported by multicrack.",
    ),
    CaseDefinition(
        "06_fibre_tensile", create_case_06, ("fibre", "fiber"),
        "experimental-smoke", "factory comments",
        "CDP-lite multicrack with fibre bridging; requires partial Numba backend.",
    ),
    CaseDefinition(
        "07_beam_4pb_jason_4pbt", create_case_07, ("jason", "4pb"),
        "experimental-unsupported", "factory + divergent archived YAML",
        "cdp_full is not faithfully supported by multicrack.",
    ),
    CaseDefinition(
        "08_beam_3pb_vvbs3_cfrp", create_case_08, ("vvbs3", "cfrp"),
        "experimental-unsupported", "factory + invalid archived YAML",
        "FRP is consumed, but cdp_full is not faithfully supported by multicrack.",
    ),
    CaseDefinition(
        "09_beam_4pb_fibres_sorelli", create_case_09, ("sorelli",),
        "synthetic-reference-unsupported", "factory; reference CSV explicitly synthetic",
        "Fibre density is canonical; cdp_full is unsupported and CSV is not experimental validation.",
    ),
    CaseDefinition(
        "10_wall_c2_cyclic", create_case_10, ("c2",),
        "experimental-unsupported", "factory + invalid archived YAML",
        "Cyclic wall definition; cdp_full is not faithfully supported by multicrack.",
    ),
    CaseDefinition(
        "11_balcony_cantilever_sls", create_case_11, ("balcony", "cantilever", "sls"),
        "experimental-unsupported", "factory comments; Eurocode SLS narrative",
        "Service trajectory is explicit; cdp_full is not faithfully supported by multicrack.",
    ),
)

CASE_REGISTRY = {definition.case_id: definition.factory for definition in CASE_DEFINITIONS}
CASE_METADATA = {definition.case_id: definition for definition in CASE_DEFINITIONS}


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for definition in CASE_DEFINITIONS:
        for alias in definition.aliases:
            key = alias.casefold()
            if key in aliases:
                raise RuntimeError(
                    f"Duplicate thesis case alias {alias!r}: "
                    f"{aliases[key]!r} and {definition.case_id!r}"
                )
            if key in CASE_REGISTRY:
                raise RuntimeError(f"Alias {alias!r} shadows a canonical case ID")
            aliases[key] = definition.case_id
    return aliases


CASE_ALIASES = _build_aliases()


@dataclass(frozen=True)
class Compatibility:
    supported: bool
    state: str
    reason: str
    numba_state: str


def evaluate_compatibility(case: CaseConfig, *, use_numba: bool) -> Compatibility:
    """Describe the real engine/material/backend combination without fallback."""
    from xfem_clean.numba.utils import NUMBA_AVAILABLE

    engine = case.solver_engine
    material = case.concrete.model_type
    features = []
    if case.rebar_layers:
        features.append("embedded bond layers")
    if case.frp_sheets:
        features.append("FRP bond layers")
    if case.fibres:
        features.append("fibre bridging")

    if engine == "auto":
        return Compatibility(
            False, "unsupported",
            "registered examples must declare solver_engine explicitly; auto dispatch is not canonical",
            "partial" if use_numba else "disabled",
        )
    if use_numba and not NUMBA_AVAILABLE:
        return Compatibility(
            False, "unsupported", "Numba was requested but is not installed",
            "unavailable",
        )
    if engine == "single" and (case.rebar_layers or case.frp_sheets):
        return Compatibility(
            False, "unsupported",
            "the canonical adapter rejects bond layers on the single-crack route",
            "partial" if use_numba else "disabled",
        )
    if engine == "single" and case.fibres:
        return Compatibility(
            False, "unsupported",
            "fibre bridging has no declared single-crack example contract",
            "partial" if use_numba else "disabled",
        )
    if engine == "multi" and material == "cdp_full":
        return Compatibility(
            False, "unsupported",
            "multicrack maps cdp_full to the CDP-lite Numba kernel; disabling Numba falls back to linear elastic assembly, so neither backend preserves cdp_full",
            "partial" if use_numba else "disabled-incompatible",
        )
    if engine == "multi" and material in {"cdp_lite", "dp"} and not use_numba:
        return Compatibility(
            False, "unsupported",
            f"multicrack {material} requires its Numba bulk kernel; the Python assembly fallback is linear elastic",
            "disabled-incompatible",
        )

    feature_text = ", ".join(features) if features else "no optional reinforcement"
    numba_state = "partial" if use_numba else "disabled"
    reason = (
        f"{engine} engine consumes {material} with {feature_text}; "
        + (
            "Numba covers constitutive/cohesive/bond kernels while mesh loops, Jacobians, sparse assembly, and linear solves remain Python/SciPy"
            if use_numba
            else "the selected elastic path has a faithful Python fallback"
        )
    )
    return Compatibility(True, "supported", reason, numba_state)


def resolve_case_id(name: str) -> str | None:
    """Resolve exact canonical IDs or aliases; partial matching is forbidden."""
    key = name.casefold()
    canonical = {case_id.casefold(): case_id for case_id in CASE_REGISTRY}
    return canonical.get(key) or CASE_ALIASES.get(key)
