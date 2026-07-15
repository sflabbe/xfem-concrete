# Canonical Case And Result Contracts

Date: 2026-07-15

Python `CaseConfig` factories are the temporary canonical case source. Divergent
YAML files are archived. `normalize_case_config()` is the sole alias and
validation boundary and raises `CaseConfigurationError` instead of selecting a
different material or solver.

The external result contract is `AnalysisResult` schema version 1. Single- and
multi-crack algorithms remain separate behind adapters. Dict compatibility is
retained while consumers migrate to canonical fields.
