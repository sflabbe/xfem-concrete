# Estado y contrato de los ejemplos

Fecha de auditoría: 2026-07-16. La fuente canónica de los 13 casos de tesis son
las factories Python de `examples/gutierrez_thesis/cases/`, registradas junto
con aliases y capacidad en `catalog.py`. Los YAML de `configs/legacy/` están
archivados, no participan en discovery y no son precedencia alternativa.

## Inventario canónico

| ID | Alias CLI principal | Fuente canónica | Config secundaria | Engine | Malla base | Steps/targets | Resultado | Estado |
|---|---|---|---|---|---:|---:|---|---|
| 01_pullout_lettow | pullout | `create_case_01` | `legacy/pullout.yml` divergente | multi | 10×10 | 100 | AnalysisResult schema 1 | operativo smoke |
| 02_sspot_frp | sspot | `create_case_02` | ninguna válida | multi | 15×15 | 100 | AnalysisResult schema 1 | operativo smoke |
| 03_tensile_stn12 | stn12 | `create_case_03` | ninguna | multi | 50×5 | 200 | AnalysisResult schema 1 | benchmark experimental soportado con Numba parcial |
| 04_beam_3pb_t5a1 | beam | `create_case_04` | ninguna | multi | 60×10 | 200 | bloqueado antes del solver | legacy activo, unsupported |
| 04a_beam_3pb_t5a1_bosco | t5a1 | `create_case_04a` | `legacy/t5a1.yml` divergente | multi | 80×16 | 200 | bloqueado antes del solver | ambiguo, unsupported |
| 04b_beam_3pb_t6a1_bosco | t6a1 | `create_case_04b` | ninguna | multi | 80×16 | 200 | bloqueado antes del solver | experimental, unsupported |
| 05_wall_c1_cyclic | c1 | `create_case_05` | ninguna | multi | 28×56 | 9 targets | bloqueado antes del solver | experimental, unsupported |
| 06_fibre_tensile | fibre | `create_case_06` | ninguna | multi | 20×20 | 200 | AnalysisResult schema 1 | smoke experimental soportado con Numba parcial |
| 07_beam_4pb_jason_4pbt | jason | `create_case_07` | `legacy/jason.yml` divergente | multi | 60×10 | 200 | bloqueado antes del solver | experimental, unsupported |
| 08_beam_3pb_vvbs3_cfrp | vvbs3 | `create_case_08` | YAML legacy inválido | multi | 86×18 | 200 | bloqueado antes del solver | experimental, unsupported |
| 09_beam_4pb_fibres_sorelli | sorelli | `create_case_09` | YAML legacy inválido | multi | 64×20 | 200 | bloqueado antes del solver | referencia sintética, unsupported |
| 10_wall_c2_cyclic | c2 | `create_case_10` | YAML legacy inválido | multi | 28×104 | 9 targets | bloqueado antes del solver | experimental, unsupported |
| 11_balcony_cantilever_sls | balcony | `create_case_11` | ninguna | multi | 80×8 | 14 targets | bloqueado antes del solver | experimental, unsupported |

`cdp_full` no se rebautiza ni se rebaja a `cdp_lite`: queda bloqueado hasta que
multicrack pueda consumir el modelo solicitado sin alterar la ley física.

## Matriz de compatibilidad real

| Familia/config | single | multicrack | legacy | Numba | FRP | bond | fibres | Estado |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| elástico sin interfaz | sí | sí | sí | parcial opcional | n/a | n/a | n/a | supported |
| elástico con rebar/FRP bond | no por adapter canónico | sí | adapter explícito | parcial opcional | sí | sí | n/a | supported |
| DP/CDP-lite | experimental | sí | adapter explícito | parcial requerido en multi | n/a | posible | sí | supported/experimental |
| CDP-full | posible en single Python, sin contrato de estos casos | no fiel | sí, congelado | multi lo empaqueta como CDP-lite | configurado | configurado | configurado | unsupported en los 9 casos registrados |
| junction enrichment | no | incompleto | n/a | n/a | n/a | n/a | n/a | unsupported |

En multicrack, Numba acelera kernels constitutivos, cohesive y bond. La
iteración sobre elementos, `det(J)`, construcción sparse, line search y solve
siguen en Python/SciPy. Por ello el CLI informa `partial`; no promete acelerar
todo el path. Desactivar Numba para DP/CDP-lite multicrack se rechaza porque el
fallback actual ensambla material elástico.

## Flujo canónico

1. Resolver ID o alias exacto, sin búsqueda parcial.
2. Cargar la factory y registrar `factory:<módulo>.<función>`.
3. Normalizar una copia schema 1.
4. Aplicar solo overrides declarados y registrar valor anterior, efectivo y opción.
5. Normalizar de nuevo y validar unidades, malla, loading, laws y output.
6. Evaluar engine/material/backend y fallar cerrado si la combinación no es fiel.
7. Construir malla y ejecutar únicamente si está soportada.
8. Devolver `AnalysisResult` schema 1 con hash SHA-256 de config y provenance.
9. El postproceso consume campos canónicos; sus dos objetos internos legacy se
   obtienen mediante `to_legacy_dict()` de forma explícita.

## Presets y modos de validación

| Preset | Factor de malla | Steps/target | Tolerancias | Uso | Numba |
|---|---:|---|---|---|---|
| coarse | 0,5 | conserva exactamente la definición | conserva exactamente la definición | caracterización; no es smoke por sí solo | parcial cuando procede |
| medium/default | 1,0 | referencia de factory | referencia de factory | benchmark completo | parcial cuando procede |
| fine | 1,5 | referencia de factory | referencia de factory | estudio de malla lento | parcial cuando procede |
| very_fine | 2,0 | referencia de factory | referencia de factory | benchmark manual | parcial cuando procede |

Un smoke reduce de forma explícita target y steps en el test que lo invoca; no
se presenta como curva física ni modifica el preset de referencia. Los smokes
de integración cubren single sintético, multicrack, pullout/bond, FRP y fibres.
Los casos completos se mantienen bajo `slow`/`--runslow` y con timeout del
harness. Los casos unsupported se validan con dry-run y error temprano.

## T5A1

El alias `t5a1` resuelve de forma determinista a
`04a_beam_3pb_t5a1_bosco`. La factory y el YAML archivado describen una viga
4000×400×200 mm, malla 80×16, 4Ø12 inferior, 2Ø10 superior, dos bond layers,
`cdp_full`, 15 mm y 200 pasos. `coarse` produce 40×8 y conserva 15 mm/200.

La documentación local y los CSV placeholder describen en cambio una viga
1500×250×120 mm con 2Ø16 y referencias distintas. Los CSV declaran ser
sintéticos, por lo que no prueban ninguna de las dos definiciones. Sin una
fuente primaria no se cambió geometría, armadura, material, carga ni tolerancia.

El engine `auto` histórico terminaba en multicrack por bond y material. La
auditoría del adapter demostró que multicrack no preserva `cdp_full`: con Numba
usa el paquete CDP-lite y sin Numba cae a elástico. Antes del cambio, coarse
alcanzó el primer substep de 0,075 mm pero no terminó una iteración en 30 s; la
ruta 80×16 con `--use-numba` tuvo el mismo límite. El hot path Python explica
la lentitud y el mismatch de material impide interpretar la no convergencia
como resultado del benchmark físico. Ahora ambos comandos salen con código 2
antes de malla/Newton y nombran la combinación incompatible.

## Otros entrypoints encontrados

No hay notebooks ni entrypoints declarados en `pyproject.toml`. Además del
runner canónico existen:

| Grupo | Archivos | Clasificación/contrato |
|---|---|---|
| demos unitarios | `ex_rebar_heaviside_angle.py`, `ex_transverse_contact.py`, `ex_crack_coalescence_junction.py` | ejemplos documentales de componentes, sin solve canónico |
| diagnósticos de ensamblaje/bond | `diagnose_*.py`, `test_bond_*.py`, `test_*debug.py`, `test_diagonal_scaling.py`, `test_first_step_debug.py`, `test_full_assembly_debug.py`, `test_newton_zero.py` | legacy activo de desarrollo; no benchmark físico |
| pullout independientes | `pullout_*.py`, `test_pullout_quick.py`, `validation_bond_slip_pullout.py` | legacy adapter/direct-engine; algunos forman el smoke standalone |
| runners históricos | `run_beam_xfem.py`, `run_gutierrez_beam.py`, `test_nonlinear_concrete_validation.py` | legacy activo, API directa del engine; no son aliases de los 13 casos |
| herramientas de casos | `parametric/parametric_study.py`, `sensitivity/sensitivity_study_jason.py`, `scripts/run_gutierrez_matrix.py` | consumen `AnalysisResult`; heredan el fail-closed del adapter |
| plots/report | `gutierrez_plots.py`, `report/generate_appendix.py` | postproceso/documental |
| operación | `scripts/run_examples_smoke.py`, `scripts/regression_manifest.py`, `scripts/run_tests.py` | smoke, regresión schema 2 y test runner |
| CDP generator | `python -m cdp_generator.cli` | herramienta independiente, no ejemplo XFEM |

Los scripts legacy directos se conservan porque tienen consumidores y valor
diagnóstico. No participan en el registry ni se anuncian como validación de los
13 casos. Su tuple/dict histórico es el adapter legacy documentado, no el
contrato público del runner de tesis.
