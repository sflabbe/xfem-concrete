# Martillazo pre-merge: timeout y compatibilidad de ejemplos

Fecha de auditoría: 2026-07-16

Commit base: `d6714c8` (`Enhance subprocess diagnostics and strengthen regression manifest`)

Entorno: Linux `6.1.0-50-amd64` x86_64, Python 3.13.5, pytest 8.4.2.

## Resumen ejecutivo

- El test de timeout dejó de depender de que un intérprete con `site` alcance
  el payload en menos de 0,4 s. El auxiliar usa `-S -u`, publica readiness por
  un archivo independiente antes de llenar los streams, espera 120 s y el
  harness aplica un timeout real de 1,0 s. `run_process()` no fue modificado.
- Se inventariaron los 13 casos de tesis y los demás entrypoints de `examples/`
  y `scripts/`. Las factories Python son la única fuente activa; los YAML se
  conservan como definiciones legacy sin precedencia.
- Los 13 casos resuelven aliases exactos, normalizan dos veces alrededor de los
  overrides, producen una vista canónica validada y declaran `engine="multi"`.
- Cuatro definiciones tienen una ruta fiel actualmente soportada: 01, 02, 03 y
  06. Nueve solicitan `cdp_full`, que el multicrack actual no consume fielmente,
  y ahora fallan antes de construir la malla con un diagnóstico determinista.
- T5A1 ya no entra en una simulación costosa o engañosa. Su definición además es
  ambigua respecto de la documentación local, por lo que no se cambió ningún
  dato físico para intentar convergencia.
- Numba se describe como backend parcial: acelera kernels constitutivos,
  cohesive y bond, pero no el loop de elementos, Jacobianos, ensamblaje sparse,
  line search ni solve.

Estado de merge: **listo con limitación documentada**. La limitación es el
soporte pendiente de `cdp_full` en multicrack; las nueve definiciones afectadas
son visibles, validables y fail-closed, no simulaciones pretendidamente válidas.

## Baseline

### Entorno y arranque

`PYTHONPATH` y `PYTHONWARNINGS` no estaban definidos. El `sitecustomize`
descubierto fue `/home/sebastian/anaconda3/lib/python3.13/sitecustomize.py`; no
había `usercustomize`. Se midieron diez arranques de cada variante:

| Invocación | mínimo (s) | mediana (s) | máximo (s) |
|---|---:|---:|---:|
| `python -c "pass"` | 0,0153 | 0,0174 | 0,0211 |
| `python -S -c "pass"` | 0,0095 | 0,0107 | 0,0116 |
| `python -u -c "pass"` | 0,0141 | 0,0152 | 0,0191 |
| `python -S -u -c "pass"` | 0,0095 | 0,0105 | 0,0117 |

El timeout de 1,0 s es aproximadamente 85 veces el peor arranque observado de
la variante elegida y sigue siendo 120 veces menor que la espera del payload.
La máquina local no reprodujo el fallo intermitente: el baseline de
`tests/test_process_utils.py` dio 9 passed en 7,88 s. La carrera sí estaba
presente en el contrato: el reloj de 0,4 s comenzaba antes de que un intérprete
con `site` pudiera crear readiness.

### Suite y T5A1 antes del cambio

- Collection baseline: 247 tests.
- Suite rápida baseline: 218 passed, 29 deselected, 18,73 s.
- Manifest schema 2: 15 passed; SHA semántico
  `d3cb0d32ade091e7390a711ca705636cb2730c5ac2097348987cc496af60773c`.
- `t5a1 --mesh coarse --no-post`: timeout exterior de 30 s, exit 124; llegó
  solamente al substep `0 -> 0.075 mm`, `ncr=0`.
- `t5a1 --use-numba --no-post`: timeout exterior de 30 s, exit 124; en malla
  80×16 tampoco terminó el primer incremento.
- En ambos intentos se observaron dos bond layers y no quedaron procesos
  residuales después del timeout exterior.

## Inventario de ejemplos

| ID | Alias CLI | Definición canónica | Fuente config | Engine | Malla | Steps | Resultado | Estado |
|---|---|---|---|---|---:|---:|---|---|
| 01_pullout_lettow | pullout | `create_case_01` | factory; YAML legacy divergente | multi | 10×10 | 100 | AnalysisResult | operativo/smoke |
| 02_sspot_frp | sspot | `create_case_02` | factory | multi | 15×15 | 100 | AnalysisResult | operativo/smoke |
| 03_tensile_stn12 | stn12 | `create_case_03` | factory | multi | 50×5 | 200 | AnalysisResult | benchmark experimental |
| 04_beam_3pb_t5a1 | beam | `create_case_04` | factory | multi | 60×10 | 200 | error temprano | legacy activo unsupported |
| 04a_beam_3pb_t5a1_bosco | t5a1 | `create_case_04a` | factory; YAML legacy divergente | multi | 80×16 | 200 | error temprano | ambiguo unsupported |
| 04b_beam_3pb_t6a1_bosco | t6a1 | `create_case_04b` | factory | multi | 80×16 | 200 | error temprano | experimental unsupported |
| 05_wall_c1_cyclic | c1 | `create_case_05` | factory | multi | 28×56 | 9 targets | error temprano | experimental unsupported |
| 06_fibre_tensile | fibre | `create_case_06` | factory | multi | 20×20 | 200 | AnalysisResult | experimental/smoke |
| 07_beam_4pb_jason_4pbt | jason | `create_case_07` | factory; YAML legacy divergente | multi | 60×10 | 200 | error temprano | experimental unsupported |
| 08_beam_3pb_vvbs3_cfrp | vvbs3 | `create_case_08` | factory; YAML legacy inválido | multi | 86×18 | 200 | error temprano | experimental unsupported |
| 09_beam_4pb_fibres_sorelli | sorelli | `create_case_09` | factory; YAML legacy inválido | multi | 64×20 | 200 | error temprano | referencia sintética unsupported |
| 10_wall_c2_cyclic | c2 | `create_case_10` | factory; YAML legacy inválido | multi | 28×104 | 9 targets | error temprano | experimental unsupported |
| 11_balcony_cantilever_sls | balcony | `create_case_11` | factory | multi | 80×8 | 14 targets | error temprano | experimental unsupported |

El inventario extendido de demos de componentes, diagnósticos, pullout legacy,
runners históricos, herramientas paramétricas y postproceso está en
`docs/examples.md`. No se encontraron notebooks ni entrypoints de packaging.

## Contrato canónico

### Fuente de verdad y precedencia

1. El catálogo resuelve un ID completo o alias exacto y rechaza búsquedas
   parciales, desconocidas, duplicadas o que oculten un ID.
2. La factory Python es la definición activa y deja provenance
   `factory:<module>.<function>`.
3. Se normaliza schema 1 antes de aplicar overrides.
4. Solo se aplican overrides CLI declarados; se registran origen, valor previo
   y valor efectivo.
5. Se vuelve a normalizar y se validan geometría, Q4, malla, loading, laws,
   fibras, orientaciones y output.
6. La matriz engine/material/backend se evalúa antes de malla y Newton.
7. Una ejecución soportada devuelve `AnalysisResult` schema 1 con hash SHA-256
   canónico y provenance; el postproceso usa `to_legacy_dict()` solo en dos
   puntos internos que requieren el adapter histórico.

Los YAML en `configs/legacy/` no se buscan ni pueden reemplazar una factory por
orden de filesystem. La prueba de paridad compara estructuras normalizadas y
reporta diferencias: `pullout.yml` tiene 26, `t5a1.yml` 5 y `jason.yml` 31; los
otros tres YAML archivados son inválidos bajo el contrato estricto.

### Matriz de compatibilidad

| Configuración | single | multicrack | legacy | Numba | FRP | bond | fibres | Estado |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| elástico sin interfaz | sí | sí | sí | parcial opcional | n/a | n/a | n/a | supported |
| elástico con bond/FRP | no en adapter canónico | sí | adapter explícito | parcial opcional | sí | sí | n/a | supported |
| DP/CDP-lite | experimental | sí | adapter explícito | parcial requerido en multi | n/a | posible | sí | supported/experimental |
| CDP-full | posible en single Python, no contratado por estos casos | no fiel | congelado | multi lo empaqueta como CDP-lite | configurado | configurado | configurado | unsupported |
| junction enrichment | no | incompleto | n/a | n/a | n/a | n/a | n/a | unsupported |

No existe fallback automático entre engines ni de material. `--no-numba` sobre
DP/CDP-lite multicrack se rechaza porque el fallback actual ensambla el bulk
elástico. `--use-numba` muestra estado `partial` y su alcance real.

## T5A1

### Trazabilidad de la definición

- Alias: `t5a1` → `04a_beam_3pb_t5a1_bosco`.
- Fuente activa: `create_case_04a`; YAML archivado `legacy/t5a1.yml`.
- Geometría factory/YAML: 4000×400×200 mm.
- Malla: 80×16; `coarse` aplica factor 0,5 → 40×8.
- Carga: control monotónico de desplazamiento, `umax=15 mm`, 200 steps;
  `coarse` no altera target, steps ni tolerancias.
- Armadura: 4Ø12 inferior a y=35 mm y 2Ø10 superior a y=355 mm.
- Bond: ambas capas tienen DOFs, filas y mappings separados. Los valores
  observados, EA=90,5 MN/perímetro=150,8 mm y EA=31,4 MN/perímetro=62,8 mm,
  corresponden a sus cantidades y diámetros; no hay evidencia de duplicación.
- Material: `cdp_full`; cracking/engine: multicrack explícito.
- Solver: 50 iteraciones máximas y substepping; no se alteraron tolerancias,
  mínimo incremento ni control.
- Postproceso: `AnalysisResult` canónico con adapter legacy explícito cuando
  procede.

### Causa raíz y decisión

`solver_interface` convertía `cdp_full` al modelo bulk `cdp`. Multicrack llama
siempre a `pack_bulk_params()`, donde `cdp`, `concrete` y `cdp-lite` seleccionan
el mismo `bulk_kind=3`, el kernel declarado CDP-lite. Con Numba desactivado no
se crea material Python para ese `bulk_kind`; el ensamblaje cae a
`LinearElasticPlaneStress`. Por tanto:

- con Numba, T5A1 no ejecutaba la ley `cdp_full` solicitada;
- sin Numba, ejecutaba bulk elástico;
- la no convergencia observada en `ncr=0` no puede caracterizarse como fallo
  físico del benchmark porque ninguna ruta respeta su material;
- el loop Python de elementos, `np.linalg.det(J)`, sparse assembly, line search
  y solve explica que el primer incremento sea costoso incluso con Numba.

La documentación/CSV local agrava la ambigüedad: describe 1500×250×120 mm y
2Ø16, mientras factory/YAML describen 4000×400×200 mm y 4Ø12+2Ø10. Los CSV se
declaran sintéticos. Sin fuente primaria no se eligió arbitrariamente una de
las dos geometrías ni se cambió física.

Decisión: T5A1 full queda **ambiguo y unsupported** hasta que exista una ruta
multicrack fiel a `cdp_full` y se resuelva la procedencia geométrica. Su smoke
es el error temprano determinista: valida definición, engine y capacidad, y
sale con código 2 antes de malla/Newton. `--use-numba` también sale con código 2
y explica que la cobertura sería parcial y no corrige la incompatibilidad.

## Cambios aplicados

| Archivo/símbolo | Cambio |
|---|---|
| `tests/test_process_utils.py` | auxiliar `-S -u`, readiness temprano independiente, timeout 1,0 s, espera 120 s; assertions intactas |
| `examples/gutierrez_thesis/catalog.py` | catálogo de 13 casos, aliases deterministas, metadata y matriz fail-closed |
| `examples/gutierrez_thesis/cases/*.py` | `solver_engine="multi"` explícito en las 13 factories |
| `case_config.py` | preserva provenance y valida unidades, geometría, malla, carga, laws, fibras y output |
| `run.py` | flujo canónico, dry-run real, overrides auditables, vista efectiva, códigos de salida y guardas antes del solver |
| `solver_interface.py` | misma guarda de compatibilidad en API directa y provenance/hash en `AnalysisResult` |
| `src/xfem_clean/results.py` | soporte explícito de provenance al adaptar bundles |
| `postprocess_comprehensive.py` | consumo canónico; legacy view solo mediante adapter explícito |
| `parametric_study.py`, `sensitivity_study_jason.py`, `run_gutierrez_matrix.py` | consumidores migrados a campos de `AnalysisResult` |
| `run_info.py` | reporte correcto de CDP-lite y cobertura parcial Numba |
| `run_examples_smoke.py` | dry-run de los 13 y smokes standalone acotados |
| `tests/test_example_contracts.py` | aliases, catálogo, normalización, paridad, engine, Numba, T5A1 y cuatro smokes reales |
| tests legacy de ejemplos | expectativas ajustadas al fail-closed y al resultado canónico |
| `README.md`, `docs/examples.md`, legacy README | estado, presets, inventario, Numba y limitaciones documentados |

No se modificaron `run_process()`, fórmulas, leyes constitutivas, Newton,
geometrías, armaduras, bond, cargas, tolerancias ni datos golden.

## Tests

Las garantías nuevas tienen nombres pequeños y separados:

- `test_catalog_has_exactly_13_cases_and_unique_outputs`
- `test_registered_case_declares_engine_and_classification`
- `test_short_aliases_and_full_names_are_deterministic`
- `test_duplicate_aliases_are_rejected`
- `test_unknown_and_partial_aliases_are_rejected`
- `test_all_registered_cases_pass_canonical_dry_run`
- `test_loadable_legacy_yaml_has_documented_normalized_diff`
- `test_t5a1_dry_run_reports_explicit_engine_and_coarse_preset`
- `test_t5a1_solve_fails_before_mesh_or_newton`
- `test_t5a1_numba_is_not_silently_ignored`
- `test_t5a1_failure_leaves_no_owned_processes`
- `test_direct_adapter_cannot_bypass_fail_closed`
- `test_cdp_lite_multicrack_rejects_unfaithful_python_fallback`
- `test_supported_family_smoke_returns_canonical_result` (01, 02, 03, 06)

Los smokes ejecutan un incremento real con target y malla reducidos de forma
explícita, verifican tipo, schema, shapes, finitud, provenance y hash. No se
presentan como validación de una curva completa.

Comandos de verificación:

```bash
.venv/bin/python -m compileall -q src examples scripts tests
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest -q tests/test_process_utils.py
.venv/bin/python -m pytest -q tests/test_regression_cases.py
.venv/bin/python -m pytest -q -m "not slow"
.venv/bin/python -m pytest -q --runslow -rs
.venv/bin/python -m examples.gutierrez_thesis.run --list
.venv/bin/python -m examples.gutierrez_thesis.run --case all --dry-run
.venv/bin/python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse --no-post
.venv/bin/python -m examples.gutierrez_thesis.run --case t5a1 --use-numba --no-post
```

El archivo completo de process utils se ejecutó diez veces: 90 tests pasaron,
sin residuales. Cada repetición mantuvo cobertura de stdout, stderr, volumen
moderado de ambos streams, Unicode, `TimeoutExpired`, hijo, nieto, cleanup
idempotente, SIGTERM y SIGINT.

Resultados finales: 274 tests collected; 245 passed y 29 deselected en la
suite rápida (20,62 s); 269 passed y 5 skipped con `--runslow` (49,45 s). Los
skips son específicos: uno para el rebuild de DOFs aún incompleto de junction
enrichment y cuatro validaciones de curvas por no estar instalada la dependencia
opcional `pandas`. No hay skips genéricos añadidos por este cambio.

## Comparación numérica

El manifest schema 2 no cambió de hash. La tabla usa exactamente sus probes;
por ello antes y después son iguales. Los cambios de hash de configuración de
las factories se deben únicamente a reemplazar `solver_engine="auto"` por
`"multi"`, explicitando el dispatch que ya ocurría, y no a valores físicos.

| Caso/probe | Antes | Después | Diferencia abs. | Diferencia rel. | Tolerancia | Estado |
|---|---:|---:|---:|---:|---:|---|
| single load (N) | 2552,1731993725325 | 2552,1731993725325 | 0 | 0 | rtol 1e-8, atol 1e-12 | igual |
| multicrack load (N) | 140,97910914191064 | 140,97910914191064 | 0 | 0 | rtol 1e-8, atol 1e-12 | igual |
| pullout bond stress (Pa) | 9675859,46729204 | 9675859,46729204 | 0 | 0 | manifest | igual |
| FRP stress (Pa) | 1000000 | 1000000 | 0 | 0 | manifest | igual |
| fibres traction (Pa) | 622046,7580187558 | 622046,7580187558 | 0 | 0 | manifest | igual |
| cyclic max target (m) | 1e-5 | 1e-5 | 0 | 0 | manifest | igual |
| SHA semántico | `d3cb0d32…` | `d3cb0d32…` | n/a | n/a | exacta | igual |

No se regeneraron curves ni golden data. T5A1 no tiene comparación numérica
válida porque el baseline no ejecutaba la ley solicitada y sus CSV son
sintéticos.

## Rendimiento

| Caso | Modo | Antes | Después | Entorno | Observación |
|---|---|---:|---:|---|---|
| timeout subprocess | archivo completo | 7,88 s | ~9,9 s | Python 3.13.5 | el segundo adicional es deliberado; 10/10 estable |
| T5A1 | coarse full | >30 s | 0,71 s | mismo host | antes quedó en primer substep; ahora incompatibilidad temprana |
| T5A1 | default + Numba | >30 s | 0,77 s | mismo host | Numba parcial no oculta material incompatible |
| 13 casos | dry-run | no existía | ~0,75 s | mismo host | resolución, normalización y capacidad, sin solver |
| case03 | smoke 1 step, 0,25 mesh | no comparable | ~3,1 s | mismo host | incluye una división adaptativa |
| case01/02/06 | smoke 1 step, 0,25 mesh | no comparable | 0,08–0,20 s | mismo host | resultados finitos y canónicos |

No se afirma speedup Numba: no se realizó una comparación representativa de
warm-up y steady-state, y los hot paths de ensamblaje continúan en Python.

## Riesgos residuales

1. Multicrack necesita una implementación/adapter fiel de `cdp_full` antes de
   habilitar los nueve benchmarks bloqueados. Eso queda fuera de este PR.
2. T5A1 requiere una fuente primaria que resuelva la discrepancia entre
   factory/YAML y documentación/CSV sintéticos.
3. Los entrypoints legacy que llaman engines directamente conservan su adapter
   histórico; están inventariados, pero no se migraron sin conocer consumidores
   externos.
4. Numba sigue siendo parcial. Una optimización del ensamblaje multicrack sería
   otro proyecto y no es necesaria para que el contrato sea honesto.

## Estado de los criterios de aceptación

| # | Estado | Evidencia |
|---:|---|---|
| 1 | cumple | process utils 10/10 |
| 2 | cumple | `-S -u`, timeout 1,0 s y readiness independiente |
| 3 | cumple | checks de hijo/nieto y residual; señales preservadas |
| 4 | cumple | catálogo/listado exacto de 13 |
| 5 | cumple | dry-run canónico de los 13; unsupported explícito |
| 6 | cumple | resolver exacto y tests de duplicados/desconocidos |
| 7 | cumple | factories activas; YAML legacy aislado y diff estructural |
| 8 | cumple | engine multi explícito y matriz por caso |
| 9 | cumple | incompatibilidades y Numba no se ignoran |
| 10 | cumple | Numba reporta `partial` o se rechaza |
| 11 | cumple | runner y consumidores operativos usan AnalysisResult/adapters explícitos |
| 12 | cumple | mismatch `cdp_full`/CDP-lite/elástico y ambigüedad documentados |
| 13 | cumple | smoke determinista de error temprano, sin hang |
| 14 | cumple | T5A1 full = ambiguo/unsupported con evidencia |
| 15 | cumple | suite rápida verde |
| 16 | cumple | suite completa verde; skips con razón específica |
| 17 | cumple | hash semántico schema 2 sin cambios |
| 18 | cumple | probes físicos idénticos; motor numérico intacto |
| 19 | cumple | `docs/examples.md`, README y catálogo concordantes |
| 20 | cumple | `git diff --check` y patch verificado con `git apply --check` |
| 21 | cumple | ZIP verificado con `unzip -t` |

Los hashes SHA-256 del patch y ZIP se entregan junto a los artefactos para
evitar una referencia circular dentro de archivos que contienen este reporte.
