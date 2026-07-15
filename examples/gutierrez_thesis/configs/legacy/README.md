# Archived configurations

These files were removed from the public configuration path on 2026-07-15.
They have no in-repository consumers and diverge from the canonical Python
factories. They are retained as historical evidence only.

| File | Status against schema/factory |
|---|---|
| `jason.yml` | Loads, but differs from the factory in 18 fields. |
| `pullout.yml` | Loads after alias normalization, but differs in 25 fields. |
| `t5a1.yml` | Loads, but differs in 4 fields. |
| `sorelli.yml` | Invalid legacy fibre structure (`fibre` block missing). |
| `vvbs3.yml` | Invalid legacy FRP structure (`bonded_length` missing). |
| `wall_c2.yml` | Invalid legacy cyclic structure (`targets` missing). |

Do not repair these by guessing values. A future migration requires source
provenance and an explicit comparison against the corresponding factory.
