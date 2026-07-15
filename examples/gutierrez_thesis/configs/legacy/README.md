# Archived configurations

These files were removed from the public configuration path on 2026-07-15.
They have no in-repository consumers and diverge from the canonical Python
factories. They are retained as historical evidence only.

| File | Status against schema/factory |
|---|---|
| `jason.yml` | Loads, but differs from the normalized factory in 31 leaf fields. |
| `pullout.yml` | Loads after alias normalization, but differs in 26 leaf fields. |
| `t5a1.yml` | Loads, but differs in 5 leaf fields. |
| `sorelli.yml` | Invalid legacy fibre structure (`fibre` block missing). |
| `vvbs3.yml` | Invalid legacy FRP structure (`bonded_length` missing). |
| `wall_c2.yml` | Invalid legacy cyclic structure (`targets` missing). |

Do not repair these by guessing values. A future migration requires source
provenance and an explicit comparison against the corresponding factory.
`tests/test_example_contracts.py` computes structural normalized diffs and
locks these counts. The extra difference introduced by the compatibility
audit is `solver_engine`: factories now declare `multi`; archived YAML keeps
the historical `auto` default and remains outside case discovery.
