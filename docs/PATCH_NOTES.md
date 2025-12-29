# XFEM v14 patch: Newton convergence fix + Reinhardt cohesive law

Este parche hace dos cosas:

1) **Arregla el bucle infinito de substepping** (y los `re-solve failed(...)` después de iniciar una grieta), corrigiendo la lógica numérica de convergencia del Newton.
2) (Opcional) añade **ley cohesiva Reinhardt/Gutiérrez** seleccionable por CLI.

---

## 1) Síntoma típico

En el paso donde se inicia una grieta:

- aparece `RuntimeError: Substepping exceeded max_subdiv=...` o
- aparece repetidamente `re-solve failed(stagnated)` / `re-solve failed(maxit)` y el substepping reduce `du` hasta valores ridículos.

Esto no es “física mala”, sino un **criterio de convergencia mal escalado**.

---

## 2) Causa raíz (2 bugs numéricos)

### Bug A — Tolerancia relativa mal definida

El código estaba usando como referencia

- `ref = ||r_f||` (residuo *actual*)

y a la vez

- `res = ||rhs|| = ||r_f||`.

Entonces la condición

- `res < tol_abs + tol_rel * ref`

se convierte (cuando `res > 1`) aproximadamente en

- `res < tol_abs`.

O sea: la parte “relativa” se cancela y terminas exigiendo una tolerancia **absoluta** extremadamente estricta (micro‑Newton), lo que después del agrietamiento es prácticamente imposible. Resultado: el Newton no declara convergencia y terminas en `maxit` o en estancamiento.

### Bug B — Escalado incorrecto de `newton_tol_du` (el “Bug #2” de QUICKFIX)

En tu versión parcheada anterior, el estancamiento/convergencia estaba escalado por el desplazamiento impuesto (`u_scale`), haciendo que con substepping (`du` muy chico) el criterio se vuelva absurdo.

---

## 3) Qué cambia el parche

### 3.1 Newton: criterio de convergencia estilo Gutiérrez (recomendado)

Gutiérrez usa el criterio (Tesis, Eq. 4.59):

> \|R\| / \|F_ext\| \le \beta

En control por desplazamiento, el “\|F_ext\|” útil es la **reacción** en el grado de libertad cargado.

**Cambio implementado** (single y multicrack):

- se calcula `P_est` como reacción en el dof cargado (o suma de dofs cargados),
- se usa `tol = newton_tol_r + newton_beta * |P_est|`,
- converge si `||R_free|| < tol`.

**Default**: `newton_beta = 1e-3` (0.1%), tal como se sugiere en la tesis.

### 3.2 Estancamiento por `||du||`: ahora es ABSOLUTO (sin `u_scale`)

Se elimina el escalado por desplazamiento impuesto y se usa:

- `if ||du|| < newton_tol_du: stagnated`.

Esto evita la “espiral de muerte” cuando el paso se hace muy chico.

---

## 4) Nueva ley cohesiva Reinhardt/Gutiérrez (opcional)

### 4.1 Selección por CLI

Se agrega:

- `--cohesive-law bilinear|reinhardt`

y parámetros:

- `--reinhardt-c1` (default 3.0)
- `--reinhardt-c2` (default 6.93)
- `--reinhardt-wcrit-mm` (si <=0, se calcula para que el área sea Gf)
- `--kcap-factor` (cap de rigidez secante: `k_sec <= kcap_factor*Kn`)

### 4.2 Implementación coherente con penalización

La ley Reinhardt se implementa como **rama de softening** después de la etapa elástica:

- 0 ≤ |δ| ≤ δ0: `t = Kn*δ`
- |δ| > δ0: softening en función de `w = |δ| - δ0`

Esto es consistente con el “elastic stage” de un cohesive law clásico y evita que la curva empiece en `t=ft` en δ=0 (que era inconsistente con la penalización).

---

## 5) Cómo aplicar

### Opción A: copiar 2 archivos

Reemplaza en tu repo:

- `xfem_xfem.py`
- `run_gutierrez_beam.py`

por los que vienen en este zip.

### Opción B: usar el script

Desde donde descomprimiste el zip:

```bash
bash apply_patch.sh /home/sebastian/xfem
```

Te crea backups `*.bak_YYYYMMDD_HHMMSS` antes de sobreescribir.

---

## 6) Ejemplos de corrida

Bilinear (default):

```bash
python -m run_gutierrez_beam --umax-mm 10 --nsteps 30 --nx 120 --ny 20 \
  --crack-mode option2 --max-cracks 8
```

Reinhardt:

```bash
python -m run_gutierrez_beam --umax-mm 10 --nsteps 30 --nx 120 --ny 20 \
  --crack-mode option2 --max-cracks 8 --cohesive-law reinhardt
```

Si quieres apretar/aflojar convergencia:

- más estricto: `--newton-beta 5e-4`
- más relajado: `--newton-beta 2e-3`

---

## 7) Archivos incluidos

- `xfem_xfem.py` (solver fixes + reinhardt)
- `run_gutierrez_beam.py` (flags nuevos)
- `xfem_xfem.diff`, `run_gutierrez_beam.diff` (unified diffs vs v13_FINAL)
- `apply_patch.sh`
