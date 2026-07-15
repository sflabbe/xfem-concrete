# Merge blockers report: manifest portable y lifecycle de subprocess

Fecha de verificación: 2026-07-15  
Baseline: `1b84c633e547c9660ab02325a72cb930f7d8b9df` (`main`, árbol limpio)  
Alcance: exclusivamente regresión numérica portable y ownership de subprocess del harness.

## Causa raíz

### Manifest

`scripts/regression_manifest.py` comparaba los probes con `np.isclose(rtol=1e-8,
atol=1e-15)`, pero además calculaba `semantic_sha256` sobre el JSON de los floats
completos. El test exigía simultáneamente ambos contratos. Una variación aceptada
por la comparación semántica cambiaba el SHA-256 y fallaba. Había un segundo hash
binario sobre un CSV cuyo float final y newline dependían del entorno. Con NumPy
2.x, el uso directo de `np.trapz()` también emitía un `DeprecationWarning`.

### Subprocess

`tests/process_utils.py` creaba una sesión nueva y solo limpiaba el grupo dentro
de la rama `TimeoutExpired`. Si pytest recibía `SIGTERM` o `SIGINT` antes del
timeout interno, no existían registro, handler ni defensa `atexit`; el hijo en la
sesión separada sobrevivía. La prueba baseline aislada reprodujo el problema: el
hijo PID 9 siguió vivo tras el `SIGTERM` exterior y luego fue eliminado por su
PGID exacto.

## Diseño implementado

### Representación canónica, schema 2

Se eligió una única estrategia: el hash cubre solamente datos discretos y el
contrato de comparación numérica; los valores numéricos se comparan por separado.
Esto corresponde a la estrategia 3 permitida y elimina cualquier contradicción
entre SHA-256 y tolerancias.

El material del hash incluye:

- `schema_version`, provenance y política de canonicalización;
- probes discretos exactos;
- mapa ordenado `caso.métrica -> tolerancia nombrada`;
- valores de `rtol` y `atol`.

La canonicalización usa JSON, UTF-8, LF, campos lexicográficos y registros
ordenados por su valor canónico. Los floats del contrato usan 15 dígitos
significativos, sin `repr(float)` como contrato; `-0.0` se normaliza a `0`, y
`NaN`, `+Inf` y `-Inf` usan tokens explícitos. El SHA-256 se calcula después de
esta transformación. El hash resultante es:

`d3cb0d32ade091e7390a711ca705636cb2730c5ac2097348987cc496af60773c`

El CSV reducido ya no tiene hash de bytes. Se escribe con LF explícito, se
parsea y se protegen exactamente columnas/newline/número de filas, mientras sus
valores se comparan con las mismas tolerancias nombradas que el resto del
manifest.

Tolerancias centralizadas:

| Nombre | `rtol` | `atol` | Uso |
|---|---:|---:|---|
| `solver_load` | `1e-8` | `1e-12` | cargas single/multicrack y CSV |
| `constitutive` | `1e-12` | `1e-15` | pullout, FRP y fibras |
| `displacement` | `1e-12` | `1e-15` | desplazamientos |
| `work` | `0` | `1e-15` | trabajo cíclico reducido |
| `exact_numeric` | `0` | `0` | valores configuracionales exactos |

Los errores numéricos informan caso, métrica, esperado, obtenido, diferencia
absoluta, diferencia relativa y tolerancia aplicable. No se redondea ningún
valor antes del cálculo físico.

### Ownership de procesos

El helper mantiene un registro protegido por `RLock`. En POSIX bloquea SIGINT y
SIGTERM durante `Popen`+registro en el hilo principal; desde otros hilos mantiene
el lock durante esa ventana. Cada hijo se crea en un process group propio y se
desregistra únicamente después de `communicate()`/`wait()` confirmado.

La misma rutina idempotente cubre:

1. salida normal y descendientes que intenten sobrevivir al líder;
2. timeout interno: TERM, gracia configurable, KILL, wait/reap;
3. SIGTERM/SIGINT del runner: cleanup completo y delegación al handler previo;
4. `atexit` como última defensa.

Se conservan stdout, stderr, comando y return code. `check=True` sigue elevando
`CalledProcessError`; los timeouts siguen elevando `TimeoutExpired` con salida.
Los tests auxiliares verifican códigos directos `-15` y `-2` para SIGTERM y
SIGINT respectivamente.

En Windows se crea un process group nuevo, pero la garantía fuerte de árbol
completo queda documentada como POSIX. No se usa `prctl` y no se añadió ninguna
dependencia.

## Inventario de creación de procesos

Procesos lanzados por pytest y bajo `run_process()`:

- `tests/test_case01_pullout_cli.py`;
- `tests/test_case03_tensile_cli.py`;
- `tests/test_case04a_bondslip_no_hang.py`;
- `tests/test_case04a_multicrack_saves.py`;
- `tests/test_examples_smoke.py`;
- `tests/test_runner_bulk_override.py`;
- collection anidada de `scripts/regression_manifest.py`.

La única llamada `Popen` fuera del registry durante tests está dentro del payload
aislado que crea deliberadamente un nieto para probar que el process group real
lo elimina. `scripts/run_tests.py` y `scripts/run_examples_smoke.py` conservan
sus `subprocess.run`: son launchers exteriores invocados por el usuario, no
procesos creados por la suite. No se encontraron `os.system`, multiprocessing,
shells de tests, `preexec_fn` ni `setsid` adicionales.

## Compatibilidad de dependencias

No cambió `pyproject.toml` ni ningún rango:

- Python `>=3.10,<3.14`;
- NumPy `>=1.24,<3`;
- SciPy `>=1.10,<2`;
- pytest `>=7.2,<9`;
- Numba opcional `>=0.56,<0.64`.

Se reutilizó `xfem_clean.utils.numpy_compat.trapezoid`: en NumPy 2 usa
`np.trapezoid`; en NumPy 1.24 usa `np.trapz`, donde no está deprecated. También
se sustituyeron los dos usos directos restantes en `cohesive_laws.py`. SciPy no
fue necesario y sus imports no cambiaron.

| Entorno | Python | NumPy | SciPy | pytest | Numba | Verificación |
|---|---:|---:|---:|---:|---:|---|
| mínimo disponible | 3.11.2 | 1.24.2 | 1.10.1 | 7.2.1 | 0.56.4 | collection, rápida, completa, manifest |
| reciente soportado | 3.13.5 | 2.3.5 | 1.16.3 | 8.4.2 | 0.63.1 | collection, rápida, focalizados, manifest con warnings como error |

## Resultados antes y después

| Verificación | Antes | Después |
|---|---:|---:|
| collection | 224 tests | 236 tests |
| rápida | 195 passed, 29 deselected | 207 passed, 29 deselected |
| completa | 219 passed, 5 skipped | 231 passed, 5 skipped |
| warnings propios deprecated (NumPy 2.x) | 1 (`np.trapz`) | 0 |
| huérfano tras SIGTERM exterior | reproducido | 0 residuales |

Tiempos y memoria máxima del entorno mínimo:

| Ejecución | Antes | RSS antes | Después | RSS después |
|---|---:|---:|---:|---:|
| collection | 4.43 s | 180.0 MiB | 3.74 s | 179.1 MiB |
| rápida | 35.21 s | 244.3 MiB | 18.73 s | 205.2 MiB |
| completa | 140.97 s | 277.7 MiB | 167.03 s | 244.1 MiB |

Los tiempos son observaciones, no objetivos de rendimiento. La diferencia de
duración se debe a variabilidad de JIT/cache/carga y a los nuevos tests de
señales.

## Comparación numérica

Antes y después dentro de cada entorno: diferencia cero para single,
multicrack, pullout, FRP, fibras y caso 09. Entre las dos combinaciones:

| Caso | mínimo | reciente | Diferencia abs. | Diferencia rel. | Tolerancia | Estado |
|---|---:|---:|---:|---:|---:|---|
| Single load N | 2552.1731993725325 | 2552.173199372528 | 4.5475e-12 | 1.7818e-15 | rtol 1e-8, atol 1e-12 | OK |
| Multicrack load N | 140.97910914191064 | 140.97910914191169 | 1.0516e-12 | 7.4593e-15 | rtol 1e-8, atol 1e-12 | OK |
| Pullout stress Pa | 9675859.46729204 | 9675859.46729204 | 0 | 0 | rtol 1e-12, atol 1e-15 | OK |
| FRP stress Pa | 1000000 | 1000000 | 0 | 0 | rtol 1e-12, atol 1e-15 | OK |
| Caso 09 density fibras/m² | 34300 | 34300 | 0 | 0 | exacta | OK |
| Caso 09 traction Pa | 622046.7580187558 | 622046.7580187558 | 0 | 0 | rtol 1e-12, atol 1e-15 | OK |

El schema público de `AnalysisResult` permanece en versión 1. Solo el artifact
versionado del manifest cambia de schema 1 a 2. Provenance conserva la corrección
del caso 09.

## Pruebas de ausencia de huérfanos

- timeout interno elimina hijo y nieto con identificador UUID;
- SIGTERM al runner auxiliar elimina hijo/nieto y conserva return code `-15`;
- SIGINT al runner auxiliar elimina hijo/nieto y conserva return code `-2`;
- cada prueba escanea `/proc` por su marcador único y obtiene lista vacía;
- timeout exterior de 3 s sobre el caso 03 devuelve 124 y deja lista vacía;
- interrupción exterior SIGINT del caso 03 devuelve 124 desde GNU timeout,
  pytest muestra `KeyboardInterrupt` y deja lista vacía;
- el registry queda en cero tras cada escenario.

## Tests añadidos o ajustados

Manifest:

- roundoff de las magnitudes observadas (`~1e-14` relativo) aceptado;
- divergencia `2e-8` rechazada con diagnóstico completo;
- metadata discreta rechazada y hash cambiado;
- orden de mappings/records estable;
- política de `-0.0`, NaN e infinitos;
- artifact JSON parseable y comparación semántica;
- deprecations propias capturadas.

Subprocess:

- proceso corto reaped;
- fallo conserva código y streams;
- timeout mata hijo y nieto;
- cleanup idempotente;
- integración exterior SIGTERM y SIGINT mediante la utilidad real.

## Archivos modificados

- `scripts/regression_manifest.py`: schema 2, hash canónico, tolerancias y CSV semántico.
- `tests/regression/canonical_manifest.json`: golden versionado revisado.
- `tests/test_regression_cases.py`: contrato portable y warnings.
- `src/xfem_clean/cohesive_laws.py`: integración trapezoidal compatible.
- `tests/process_utils.py`: registry, señales, timeout, reap y `atexit`.
- `tests/conftest.py`: instalación temprana del lifecycle.
- `tests/process_signal_runner.py`: runner auxiliar controlado.
- `tests/test_process_utils.py`: tests focalizados de lifecycle.
- `docs/testing.md`: contrato canónico y política POSIX.
- `docs/validation.md`: separación hash/valores numéricos.
- `MERGE_BLOCKERS_REPORT.md`: este reporte.

## Alternativas descartadas

- Hash de floats redondeados o cuantizados: todavía puede cruzar un límite de
  bucket aunque ambos valores cumplan la tolerancia.
- Mantener el hash CSV binario: reintroduce dependencia de float/newline.
- Hash de valores numéricos más comparación tolerante: mantiene dos contratos
  paralelos y contradictorios.
- Solo `atexit`: no cubre de forma suficiente las señales exteriores.
- Solo `prctl(PR_SET_PDEATHSIG)`: Linux-only y no cubre todas las rutas.
- Dependencia externa de gestión de procesos: innecesaria para este alcance.

## Comandos de verificación

```bash
python3 -m compileall -q src examples tests scripts
python3 -m pytest --collect-only -q
timeout --signal=TERM --kill-after=15s 900s python3 -m pytest -m 'not slow' -q --durations=30
timeout --signal=TERM --kill-after=15s 1200s python3 -m pytest --runslow -q --durations=40
PYTHONWARNINGS=error::DeprecationWarning PYTHONPATH=src python3 scripts/regression_manifest.py
PYTHONWARNINGS=error::DeprecationWarning PYTHONPATH=src .venv/bin/python scripts/regression_manifest.py
.venv/bin/python -m pytest -m 'not slow' -q
patch --dry-run -p1 < XFEM_MERGE_BLOCKERS.patch
unzip -t xfem-concrete-merge-blockers.zip
sha256sum XFEM_MERGE_BLOCKERS.patch xfem-concrete-merge-blockers.zip
```

## Criterios de aceptación

Los 18 criterios quedan satisfechos: compile/collection/suites pasan; no hay
warnings propios; roundoff aceptado y divergencia real rechazada; hash canónico
estable y documentado; todos los procesos de tests tienen ownership; timeout y
señales eliminan/reap; códigos correctos; no cambian resultados físicos; caso
09 sigue en 34 300 fibras/m²; no se tocó el engine numérico fuera del reemplazo
API-equivalente de trapezoidal; patch y ZIP se validan con los comandos listados.

## Riesgos residuales

- La garantía fuerte de descendientes es POSIX; Windows solo tiene process group
  nuevo y debe recibir una implementación Job Object si pasa a ser plataforma CI
  oficialmente soportada.
- SIGKILL al propio pytest o pérdida de la máquina no permite ejecutar cleanup;
  ningún handler en Python puede garantizarlo.
- La suite completa se ejecutó en la combinación mínima disponible; la reciente
  ejecutó suite rápida y tests focalizados, no la suite lenta completa.

## Próximo PR recomendado

Después de cerrar estos bloqueos, el siguiente PR independiente puede abordar
la revisión de dominio de la tangente Drucker–Prager, sin mezclarla con este
trabajo.
