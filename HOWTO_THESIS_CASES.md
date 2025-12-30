# HOWTO: Ejecutar Casos de Tesis Gutiérrez

Este documento describe cómo ejecutar los casos 01-06 de la tesis Gutiérrez después de la integración multicrack + bond-slip + postproceso (FASES D-G).

## Resumen de Cambios Implementados

### FASE D: Multicrack + Bond-slip integrado
- ✓ Dispatcher en `solver_interface.py` para seleccionar solver apropiado (single-crack/multicrack/cyclic)
- ✓ Multicrack acepta `bc_spec`, `u_targets`, `nodes`, `elems` externos
- ✓ Multicrack usa `bond_law` externo (no hardcode)
- ✓ Soporte para `perimeter_total` explícito y `segment_mask` para bond-disabled regions
- ✓ Resolución de negative markers para steel DOFs en bc_spec

### FASE G: Estados expuestos + Postproceso
- ✓ `ResultBundle` opcional en `analysis_single.py` (retorna `bond_states`, `rebar_segs`, `dofs`)
- ✓ Postproceso comprehensive conectado:
  - CSV: `load_displacement.csv`, `slip_profile_final.csv`, `bond_stress_profile_final.csv`
  - PNG: `load_displacement.png`, `slip_profile_stepXXXX.png`, `bond_stress_profile_stepXXXX.png`
  - VTK: `vtk/step_XXXX.vtk` (formato ASCII)

### FASE F (Partial): Trayectoria cíclica
- ✓ Generador de `u_targets` para loading cíclico (`generate_cyclic_u_targets`)
- ⚠️ Cyclic driver aún no implementado completamente (requiere integración a analysis_single)

### FASE E: FRP y Fibres
- ⚠️ **NO IMPLEMENTADO** en este branch (requiere DOFs FRP + t_fibre)

## Casos Soportados

| Caso | ID | Solver | Estado |
|------|----|----|--------|
| 01 Pullout (Lettow) | `pullout` | Single-crack | ✓ Funciona |
| 02 FRP Debonding | `frp` | ⚠️ No implementado | ❌ |
| 03 Tensile STN12 | `tensile` | Multicrack | ⚠️ Por probar |
| 04 Beam 3PB | `beam` | Multicrack | ⚠️ Por probar |
| 05 Wall Cyclic | `wall` | Multicrack+Cyclic | ⚠️ No implementado |
| 06 Fibre Tensile | `fibre` | ⚠️ No implementado | ❌ |

## Cómo Ejecutar

### Caso 01: Pullout (Lettow)

El caso pullout es el **único completamente testeado** y funcional con bond-slip + segment_mask.

```bash
# Ejecutar con malla gruesa (rápido)
PYTHONPATH=src python -m examples.gutierrez_thesis.run --case pullout --mesh coarse --nsteps 5

# Ejecutar con malla fina (más preciso, más lento)
PYTHONPATH=src python -m examples.gutierrez_thesis.run --case pullout --mesh fine --nsteps 20
```

**Outputs esperados** (en `outputs/case_01_pullout/`):
- `load_displacement.csv`: Curva Fuerza-Desplazamiento
- `load_displacement.png`: Gráfico P-δ
- `slip_profile_final.csv`: Perfil de deslizamiento s(x) en el último paso
- `bond_stress_profile_final.csv`: Perfil de tensión de adherencia τ(x)
- `vtk/step_XXXX.vtk`: VTK para visualización en ParaView

**Verificación física esperada**:
- Curva Load-Slip **no trivial** (debe mostrar comportamiento no-lineal típico de bond-slip)
- τ(x) ≈ 0 en x ∈ [0, 164] mm (zona bond-disabled por `bond_disabled_x_range`)
- τ(x) activo en x > 164 mm (zona bonded)

### Caso 03: Tensile STN12 (Multicrack)

**Advertencia**: Este caso usará el solver multicrack pero **NO ha sido testeado** después de la integración.

```bash
PYTHONPATH=src python -m examples.gutierrez_thesis.run --case tensile --mesh coarse --nsteps 10
```

**Comportamiento esperado**:
- Múltiples grietas distribuidas (cracking pattern)
- Postproceso generará PNG con múltiples cracks

**Si falla**, verificar:
1. `_should_use_multicrack()` en `solver_interface.py` detecta correctamente el case_id
2. BCs son compatibles (default 3PB puede no ser apropiado para tensile)

### Caso 04: Beam 3PB (Multicrack)

```bash
PYTHONPATH=src python -m examples.gutierrez_thesis.run --case beam --mesh coarse --nsteps 10
```

**Comportamiento esperado**:
- Grietas flexurales (verticales) distribuidas
- Posible transición a grietas diagonales según `crack_mode="option2"`

### Casos NO soportados (requieren FASE E)

**Caso 02 (FRP)** y **Caso 06 (Fibres)** requieren implementación adicional:
- **FRP**: Necesita DOFs FRP, bond-slip FRP sobre superficie, y BCs específicos
- **Fibres**: Necesita implementación de `t_fibre(w)` en ley cohesiva

## Estructura de Archivos

```
xfem-concrete/
├── src/xfem_clean/
│   ├── xfem/
│   │   ├── analysis_single.py     # Solver single-crack (con return_bundle)
│   │   ├── multicrack.py          # Solver multicrack (integrado)
│   │   └── ...
│   ├── bond_slip.py                # Bond-slip laws + assembly
│   └── ...
├── examples/gutierrez_thesis/
│   ├── case_config.py              # Dataclasses CaseConfig
│   ├── solver_interface.py         # Dispatcher + case_config → solver
│   ├── postprocess_comprehensive.py # Postproceso CSV/PNG/VTK
│   ├── run.py                      # CLI para ejecutar casos
│   └── cases/
│       ├── case_01_pullout_lettow.py
│       ├── case_03_tensile_stn12.py
│       └── ...
├── test_smoke_integration.py       # Tests smoke (verifican imports)
└── HOWTO_THESIS_CASES.md           # Este archivo
```

## Debug / Troubleshooting

### Error: "Multicrack integration not yet complete"

Si ves este error, significa que el dispatcher detectó que el caso requiere multicrack pero algo falló.

**Solución**:
1. Verificar que `_should_use_multicrack(case)` retorna True/False correctamente
2. Verificar que `run_analysis_xfem_multicrack` recibe todos los parámetros necesarios
3. Inspeccionar logs para ver si hay warnings de bond_law fallback

### Error: "bond_states is None"

Si el postproceso falla con `bond_states is None`, significa que:
- El caso NO tiene bond-slip habilitado (`enable_bond_slip=False`)
- O el solver no retornó `bond_states` en el bundle

**Solución**:
- Verificar `model.enable_bond_slip` en el XFEMModel
- Verificar que `return_bundle=True` está activo en la llamada a `run_analysis_xfem`

### Cyclic loading no funciona

El dispatcher detecta cyclic loading y genera `u_targets`, pero el driver cíclico **aún NO está implementado** en `analysis_single.py`.

**Workaround**: Usar multicrack con `u_targets` (solo si el caso soporta multicrack).

## Próximos Pasos (FASE E)

Para completar los casos 02 y 06, implementar:

1. **FRP (Caso 02)**:
   - Agregar DOFs FRP en `build_xfem_dofs` (similar a steel)
   - Implementar bond-slip FRP en superficie (perimeter = width_bonded)
   - BCs específicos para single-lap shear test

2. **Fibres (Caso 06)**:
   - Implementar `t_fibre(w)` en la ley cohesiva como término extra
   - Calibrar `rho_eff` desde `fibre.density` (fibres/cm²)
   - Usar BanholzerBondLaw para pullout de fibras

## Commits Realizados

1. `feat(results)`: ResultBundle opcional con bond_states/rebar_segs/dofs
2. `feat(dispatch)`: Dispatch (single/multicrack/cyclic) + u_targets generator
3. `feat(multicrack-bcs)`: Multicrack acepta bc_spec + u_targets + steel markers
4. `fix(multicrack-bond)`: Multicrack usa bond_law externo + perimeter + segment_mask
5. `feat(dispatch)`: Conectar multicrack al dispatcher
6. `feat(postprocess)`: Postproceso comprehensive conectado
7. `test(smoke)`: Tests smoke básicos

## Contacto / Issues

Si encuentras problemas, verifica:
1. Sintaxis correcta: `python -m py_compile <archivo.py>`
2. Imports funcionan: ver `test_smoke_integration.py`
3. Logs del solver (verbosidad controlada por `model.verbose`)

---
**Autor**: Claude (coding agent)
**Branch**: `claude/multicrack-bond-slip-Lzeoj`
**Fecha**: 2025-12-30
