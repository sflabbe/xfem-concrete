# Cierre final de bloqueos de merge del harness XFEM

Fecha de verificación: 2026-07-15

Baseline: `a81e594b8bd50f42fcb0fe6ac9bea8f1e1923eb1` (`main`, árbol limpio)

Alcance: captura de diagnósticos de subprocess y recolección fail-closed del manifest.

## Resumen

Los dos bloqueos quedan cerrados con un cambio focalizado en el harness. La
captura de `stdout` y `stderr` usa dos archivos temporales binarios desde el
instante de `Popen`, y `run_process()` espera con `wait()` en vez de depender del
estado interno de dos llamadas a `communicate()`. La recolección del manifest
sigue siendo parte del contrato publicado y ahora falla ante return code no cero,
salida no parseable o contador cero.

No se modificaron el motor numérico, golden data, schema 2, canonicalización,
tolerancias, provenance ni hashes. El SHA-256 semántico permanece:

`d3cb0d32ade091e7390a711ca705636cb2730c5ac2097348987cc496af60773c`

## Baseline

| Entorno | Python | NumPy | SciPy | pytest | Numba |
|---|---:|---:|---:|---:|---:|
| sistema | 3.11.2 | 1.24.2 | 1.10.1 | 7.2.1 | 0.56.4 |
| `.venv` | 3.13.5 | 2.3.5 | 1.16.3 | 8.4.2 | 0.63.1 |

Antes de editar:

- checkout limpio en `a81e594b8bd50f42fcb0fe6ac9bea8f1e1923eb1`;
- 236 tests collected en 1.82 s (2.91 s wall), RSS máxima 184108 KiB;
- focalizados existentes: 13 passed en 9.47 s, RSS máxima 183316 KiB;
- manifest válido: 8 probes, 236 tests y SHA-256 canónico sin cambios;
- no había procesos con los marcadores de control antes de las reproducciones.

El host no tiene un binario global llamado `python`. Para el comando documentado
se usó `.venv/bin/python`, que es la combinación reciente soportada; también se
verificó `python3`/pytest del sistema.

## Causa raíz 1 — pérdida de streams

### Comportamiento de `communicate()`

La rama baseline llamaba `communicate(timeout=...)`, ignoraba `output` y
`stderr` del `TimeoutExpired` original, terminaba el grupo y llamaba a
`communicate()` otra vez. Ese segundo resultado no es una fuente canónica de los
bytes ya entregados por la primera llamada: en una combinación soportada puede
contener solo lo restante y devolver dos cadenas vacías. Reconstruir una segunda
excepción exclusivamente con ese resultado pierde diagnósticos ya emitidos y
flushed.

La reproducción determinista ejecutada directamente sobre el blob de
`HEAD:tests/process_utils.py` modeló ese comportamiento soportado: la primera
excepción contenía `flushed-stdout`/`flushed-stderr`, el segundo `communicate()`
devolvió vacío y el `TimeoutExpired` reconstruido expuso `stdout == ''` y
`stderr == ''`. En el subprocess real de Python 3.13.5 de este host, la segunda
llamada volvió a entregar el contenido; también se observó que la excepción
intermedia contenía bytes pese a `text=True`. La corrección no depende de ninguna
de las dos variantes.

### Diseño elegido

Se eligió una única estrategia: archivos temporales separados para `stdout` y
`stderr`.

- Los descriptores se conectan a `Popen` antes de que el hijo empiece a ejecutar.
- Los streams se drenan continuamente a archivos regulares; no existe capacidad
  limitada de pipe que pueda bloquear al hijo.
- El orden dentro de cada stream es el orden de escritura que conserva el
  descriptor correspondiente.
- Tras salida normal o TERM/KILL/reap, cada archivo se rebobina y se decodifica
  explícitamente como UTF-8 con errores estrictos.
- La API continúa devolviendo `str`: `CompletedProcess[str]`,
  `CalledProcessError.stdout/stderr` y `TimeoutExpired.output/stdout/stderr`.
- Los dos archivos se cierran por context manager en todas las rutas y se
  eliminan automáticamente. No se crean threads y, por tanto, no hay readers que
  unir ni excepciones de thread que ocultar.

### Timeout y lifecycle

`wait(timeout)` determina exclusivamente si expiró el plazo. Al expirar se usa la
misma rutina de ownership ya existente: TERM al process group, gracia
configurable, KILL si sigue vivo, `wait()` del hijo directo y espera del grupo.
Solo después se leen ambos archivos y se levanta `TimeoutExpired` con comando y
timeout originales. El proceso permanece registrado hasta que `poll()` confirma
el reap.

La salida normal conserva return code y ambos streams. Un return code no cero no
se confunde con timeout: `check=False` lo devuelve y `check=True` sigue levantando
`CalledProcessError` con código y diagnósticos completos. SIGTERM, SIGINT,
`atexit` y cleanup idempotente siguen usando la misma registry.

### Readiness

Los tests de timeout ya no parsean PIDs desde el stream sometido a prueba. El
hijo publica readiness y PIDs mediante reemplazo atómico de un JSON temporal;
stdout y stderr quedan reservados para diagnósticos. No se usan sleeps largos
como mecanismo principal de sincronización. Cada comando de pytest se ejecutó
además con timeout exterior de GNU `timeout`.

## Causa raíz 2 — manifest fail-open

El comando interno es:

```text
<sys.executable> -m pytest --collect-only -q
```

El baseline lo ejecutaba con `check=False`, ignoraba `returncode` y buscaba con
regex un contador en stdout+stderr. Si pytest fallaba o cambiaba la salida, la
función devolvía `None`; como `environment` no forma parte de
`compare_manifests()`, el CLI podía anunciar éxito.

La reproducción sobre `HEAD:scripts/regression_manifest.py`, con
`PYTEST_ADDOPTS=--xfem-invalid-option`, obtuvo return code interno 4 y aun así
imprimió `manifest OK: 8 probes, None tests collected` y devolvió 0.

### Contrato final

Se conserva la recolección como parte del manifest y se hace fail-closed:

1. return code distinto de cero levanta `CollectionError`;
2. un resumen que no contenga una línea estable `<N> test(s) collected ...`
   levanta `CollectionError`;
3. el cero se rechaza explícitamente para este repositorio;
4. un manifest válido solo puede contener un entero positivo;
5. el CLI captura ese error, imprime diagnóstico en stderr y devuelve 2;
6. un mismatch numérico/contractual sigue devolviendo 1;
7. `manifest OK` solo se imprime tras collection y comparación exitosas.

El diagnóstico incluye comando, return code, `PYTHONWARNINGS`, stdout y stderr.
No se añadió plugin ni dependencia: pytest no ofrece aquí una salida estructurada
de collection mediante sus opciones core, por lo que se valida estrictamente su
línea resumen de terminal y se falla si no está presente.

## Archivos modificados

- `tests/process_utils.py`: captura canónica por archivos temporales, espera y
  decodificación UTF-8 tras reap.
- `tests/test_process_utils.py`: readiness independiente y cobertura de stdout,
  stderr, ambos streams, volumen, Unicode, salida normal, árbol de procesos,
  señales y cleanup.
- `scripts/regression_manifest.py`: collection fail-closed, diagnóstico completo
  y código CLI 2 para errores de collection/parsing.
- `tests/test_regression_cases.py`: tests de collection, warning fatal, parseo,
  contador y códigos CLI.
- `FINAL_MERGE_FIX_REPORT.md`: evidencia y estado final.

No se modificó ningún archivo de `src/`, golden data ni documentación del
contrato numérico.

## Tests añadidos o corregidos

### Subprocess

- `test_short_process_is_reaped`: salida normal exacta, ambos streams, Unicode y
  return code.
- `test_failed_process_preserves_return_code_and_diagnostics`: `check=True`
  conserva código y streams.
- `test_timeout_preserves_stdout_and_output_alias`: stdout UTF-8 y alias
  `output` después de timeout.
- `test_timeout_preserves_stderr`: stderr completo después de timeout.
- `test_timeout_preserves_both_streams_without_pipe_deadlock`: contenido
  distinto y exacto, 128 KiB por stream.
- `test_internal_timeout_kills_child_and_grandchild`: readiness por archivo,
  captura de ambos streams y desaparición real de ambos PIDs.
- `test_cleanup_is_idempotent`: cleanup repetido.
- `test_outer_runner_signal_kills_child_group[sigterm]`: SIGTERM, código `-15` y
  cero descendientes.
- `test_outer_runner_signal_kills_child_group[sigint]`: SIGINT, código `-2` y
  cero descendientes.

### Manifest

- `test_collection_count_parses_successful_pytest_output`: devuelve entero.
- `test_collection_nonzero_exit_preserves_diagnostics`: return code, comando y
  ambos streams se conservan.
- `test_collection_rejects_untrustworthy_count`: salida no parseable y cero
  fallan, nunca devuelven `None`.
- `test_collection_warning_promoted_to_error_is_not_swallowed`: warning fatal no
  se convierte en éxito.
- `test_cli_collection_failure_is_nonzero_and_never_prints_ok`: código 2 y sin
  falso mensaje OK.
- `test_cli_success_requires_integer_count`: código 0 solo con entero válido.
- `test_cli_comparison_failure_is_nonzero`: mismatch conserva código 1.
- `test_canonical_regression_manifest`: 8 probes, entero positivo, mismo hash y
  sin deprecations propias.
- Permanecen los guards de roundoff permitido, diferencia sobre tolerancia,
  metadata discreta, NaN/Inf, `-0.0` y roundtrip canónico.

## Verificación

| Verificación | Resultado | Tiempo pytest | Wall | RSS máxima |
|---|---|---:|---:|---:|
| compileall | OK | — | — | — |
| collection | 247 collected | 1.85 s | 3.03 s | 184208 KiB |
| focalizados, Python 3.11 | 24 passed | 11.13 s | 11.94 s | 183732 KiB |
| focalizados, Python 3.13 | 24 passed | 10.86 s | 12.02 s | 191208 KiB |
| rápida | 218 passed, 29 deselected | 17.04 s | 18.34 s | 207312 KiB |
| completa | 242 passed, 5 skipped | 147.57 s | 148.85 s | 251432 KiB |

No hubo failed, errors ni warnings en esas ejecuciones. Los cinco skips de la
completa son los ya justificados por plataforma/capacidad en el baseline.

Manifest positivo con warnings estrictos:

```text
manifest OK: 8 probes, 247 tests collected, sha256=d3cb0d32ade091e7390a711ca705636cb2730c5ac2097348987cc496af60773c
```

Resultado: código 0, 4.59 s wall, RSS máxima 181116 KiB.

Manifest negativo controlado:

- `PYTEST_ADDOPTS=--xfem-invalid-option`;
- pytest interno: código 4;
- CLI del manifest: código 2;
- stdout/stderr y warnings estrictos presentes en el diagnóstico;
- ausencia de `manifest OK`;
- 2.51 s wall, RSS máxima 181024 KiB;
- checkout sin cambios adicionales después de la prueba.

La ejecución aislada de `tests/test_process_utils.py` terminó con 9 passed en
7.81 s. Sus asserts verifican SIGTERM, SIGINT, PIDs de hijo/nieto y registry en
cero. Un escaneo posterior de `/proc` excluyendo ancestros devolvió
`residual_marker_processes=[]`.

## Comparación numérica

| Caso | Antes | Después | Diferencia abs. | Diferencia rel. | Tolerancia | Estado |
|---|---:|---:|---:|---:|---:|---|
| Single, load N | 2552.1731993725325 | 2552.1731993725325 | 0 | 0 | rtol 1e-8, atol 1e-12 | OK |
| Multicrack, load N | 140.97910914191064 | 140.97910914191064 | 0 | 0 | rtol 1e-8, atol 1e-12 | OK |
| Pullout, stress Pa | 9675859.46729204 | 9675859.46729204 | 0 | 0 | rtol 1e-12, atol 1e-15 | OK |
| FRP, stress Pa | 1000000 | 1000000 | 0 | 0 | rtol 1e-12, atol 1e-15 | OK |
| Caso 09, fibras/m² | 34300 | 34300 | 0 | 0 | exacta | OK |
| Caso 09, traction Pa | 622046.7580187558 | 622046.7580187558 | 0 | 0 | rtol 1e-12, atol 1e-15 | OK |

`compare_manifests()` devolvió lista vacía. `AnalysisResult` permanece en schema
1; provenance y tolerancias son exactamente iguales; el manifest conserva 8
probes y el mismo SHA-256.

## Comandos exactos

```bash
git status --short
git rev-parse HEAD
python3 --version
python3 -c "import numpy, scipy, pytest, numba; print(numpy.__version__, scipy.__version__, pytest.__version__, numba.__version__)"
.venv/bin/python --version
.venv/bin/python -c "import numpy, scipy, pytest, numba; print(numpy.__version__, scipy.__version__, pytest.__version__, numba.__version__)"

python3 -m compileall -q src examples tests scripts
pytest --collect-only -q
timeout --signal=TERM --kill-after=5s 90s pytest -q tests/test_process_utils.py tests/test_regression_cases.py
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m pytest -q tests/test_process_utils.py tests/test_regression_cases.py
timeout --signal=TERM --kill-after=15s 900s /usr/bin/time -v pytest -q -m "not slow" --durations=30
timeout --signal=TERM --kill-after=15s 1200s /usr/bin/time -v pytest -q --runslow --durations=40

PYTHONWARNINGS=error::DeprecationWarning PYTHONPATH=src .venv/bin/python scripts/regression_manifest.py
PYTEST_ADDOPTS=--xfem-invalid-option PYTHONWARNINGS=error::DeprecationWarning PYTHONPATH=src .venv/bin/python scripts/regression_manifest.py

pytest -q tests/test_process_utils.py
git diff --check
git apply --check XFEM_FINAL_MERGE_FIX.patch
unzip -t xfem-concrete-final-merge-fix.zip
sha256sum XFEM_FINAL_MERGE_FIX.patch xfem-concrete-final-merge-fix.zip
```

Las reproducciones baseline se ejecutaron desde los blobs exactos obtenidos con
`git show HEAD:scripts/regression_manifest.py` y
`git show HEAD:tests/process_utils.py`, sin restaurar ni modificar el checkout.

## Criterios de aceptación

| # | Criterio | Estado |
|---:|---|---|
| 1 | Test original pasa sin debilitarse | OK; readiness se fortaleció |
| 2 | Timeout conserva stdout | OK |
| 3 | Timeout conserva stderr | OK |
| 4 | Ambos streams | OK |
| 5 | Sin deadlock por pipes | OK; 128 KiB por stream |
| 6 | Hijo y nieto eliminados/reaped | OK |
| 7 | Cero residuales | OK |
| 8 | Cleanup SIGTERM | OK, return code `-15` |
| 9 | Cleanup SIGINT | OK, return code `-2` |
| 10 | Manifest falla con return code no cero | OK |
| 11 | Manifest falla con salida no parseable | OK |
| 12 | Nunca `tests_collected=None` válido | OK; entero positivo obligatorio |
| 13 | `manifest OK` solo tras todos los checks | OK |
| 14 | Hash canónico estable | OK |
| 15 | Cero warnings deprecated propios | OK |
| 16 | Suite rápida | OK |
| 17 | Suite completa | OK; solo 5 skips existentes |
| 18 | Resultados físicos invariantes | OK; diferencias 0 |
| 19 | Patch aplica sobre checkout limpio | OK; `git apply --check` |
| 20 | ZIP íntegro | OK; `unzip -t` |

## Riesgos residuales

- La garantía fuerte sobre todo el árbol sigue siendo POSIX. En Windows se crea
  process group, pero eliminar nietos requeriría Job Objects si esa plataforma se
  convierte en CI soportada para esta garantía.
- La captura ahora está limitada por espacio temporal disponible, no por memoria
  ni por la capacidad de un pipe. Un subprocess que produzca salida ilimitada
  puede agotar el filesystem temporal.
- La decodificación es UTF-8 estricta para mantener una API textual determinista.
  Un comando que emita bytes no UTF-8 fallará explícitamente al leer diagnósticos.
- SIGKILL al propio runner o pérdida de la máquina no permite ejecutar handlers;
  ningún cleanup en Python puede garantizar esa ruta.

## Estado de merge

Tras estas pruebas, el repositorio queda listo para merge. No permanece ningún
bloqueo conocido dentro del alcance de este micro-PR.
