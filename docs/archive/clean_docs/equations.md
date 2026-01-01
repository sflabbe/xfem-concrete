# Índice de Ecuaciones: Tesis → Código

**Mapeo completo de ecuaciones de Gutiérrez (2020) a implementación Python**

---

## 📐 Capítulo 3: Descripción del Problema

### 3.2 Modelo de Grieta Cohesiva

| Ecuación | Descripción | Módulo | Implementación |
|----------|-------------|--------|----------------|
| **3.17** | Descomposición de tracción: **t** = t_n **n̂** + t_s **t̂** | `cohesive_laws.py` | Retorno de `traction()` |
| **3.18** | Descomposición de apertura: **ω** = ω_n **n̂** + ω_s **t̂** | `cohesive_laws.py` | Parámetros de `traction()` |
| **3.24** | **Ley de Reinhardt (no-lineal):**<br/>t_n = f_t [(1 + (c₁ω/ω_c)³) exp(-c₂ω/ω_c) - (ω/ω_c)(1 + c₁³)exp(-c₂)] | `cohesive_laws.py` | `ReinhardtCohesiveLaw._traction_loading()` |
| **3.25** | **Apertura crítica:**<br/>ω_c = 5.136 G_f / f_t | `cohesive_laws.py` | `ReinhardtCohesiveLaw.__post_init__()` |
| **3.26** | Tracción de corte: t_s = d₀ exp(h_s ω_n) ω_s | `cohesive_laws.py` | `traction()` línea ~157 |
| **3.27** | Degradación de corte: h_s = ln(d₁/d₀) | `cohesive_laws.py` | `__post_init__()` línea ~147 |
| **3.28** | **Parámetro de historia:**<br/>κ(t) = max_{T≤t} ω_n(T) | `cohesive_laws.py` | Parámetro `kappa` |
| **3.29** | **Descarga/recarga lineal:**<br/>t_n = t_n,max + k(ω_n - κ) | `cohesive_laws.py` | `traction()` líneas ~156-158 |
| **3.30** | Tracción corte con historia: t_s = d₀ exp(h_s κ) ω_s | `cohesive_laws.py` | `traction()` línea ~161 |

---

### 3.2.2 Iniciación y Propagación de Grietas

| Ecuación | Descripción | Módulo | Implementación |
|----------|-------------|--------|----------------|
| **3.31** | **Esfuerzo principal máximo (Rankine):**<br/>σ_max = (σ_xx + σ_yy)/2 + √[((σ_xx - σ_yy)/2)² + σ_xy²] | `crack_criteria.py` | `principal_stress_2d()` líneas ~40-47 |
| **3.32** | **Dirección principal:**<br/>θ_I = (1/2) arctan(2σ_xy / (σ_xx - σ_yy)) | `crack_criteria.py` | `principal_stress_2d()` líneas ~50-52 |
| **3.33** | Determinación de dirección máxima | `crack_criteria.py` | `principal_stress_2d()` líneas ~55-67 |
| **3.34** | Criterio propagación local: σ_n = **n̂**ᵀ **σ** **n̂** = f_t | `crack_criteria.py` | Comentario en `NonLocalPropagationCriterion` |
| **3.35** | **Esfuerzo no-local:**<br/>σ̃ = [∫ w(r) dΩ]⁻¹ ∫ σ(**x**) w(r) dΩ | `crack_criteria.py` | `NonLocalAveraging.average_stress()` líneas ~139-168 |
| **3.36** | **Función peso Gaussiana:**<br/>w(r) = (1/(l√(2π))) exp(-r²/(4l²)) | `crack_criteria.py` | `NonLocalAveraging.weight()` líneas ~120-132 |
| **3.37** | **Criterio propagación no-local:**<br/>σ̃_n = **n̂**ᵀ σ̃ **n̂** = f_t | `crack_criteria.py` | `NonLocalPropagationCriterion.check()` líneas ~204-235 |

**Ángulo de propagación:** θ_p = θ_max + 90° (perpendicular al esfuerzo principal)

---

## 📐 Capítulo 4: Método de Elementos Finitos Extendidos

### 4.1.3 Ecuación de Equilibrio Discretizada

| Ecuación | Descripción | Módulo | Implementación |
|----------|-------------|--------|----------------|
| **4.48** | **Vector residual:**<br/>**R**ᵉ(**U**) = **K****U** + **F**_D(**U**) + **F**_coh(**U**) - **F**_ext | `newton_solver.py` | (Próximo) Ensamble de residual |
| **4.49** | Expansión lineal Newton-Raphson | `newton_solver.py` | (Próximo) Loop de Newton |
| **4.50** | **Ecuación incremental:**<br/>**K**_n Δ**U**_{n+1} = -**R**ᵉ(**U**_n) | `newton_solver.py` | (Próximo) Solve lineal |
| **4.51** | Matriz rigidez global: **K** = **K**_std + **K**_D + **K**_coh | `newton_solver.py` | (Próximo) Ensamble tangente |
| **4.52** | **Matriz cohesiva:**<br/>**K**_coh = ∂**F**_coh/∂**U** | `cohesive_laws.py` | `tangent_stiffness()` |
| **4.53** | **Rigidez tangente cohesiva:**<br/>∂**T**/∂**Ω** = [[∂t_n/∂ω_n, ∂t_n/∂ω_s], [∂t_s/∂ω_n, ∂t_s/∂ω_s]] | `cohesive_laws.py` | `tangent_stiffness()` líneas ~184-210 |
| **4.58** | **Actualización de desplazamientos:**<br/>**U**_{n+1} = **U**_n + Δ**U**_{n+1} | `newton_solver.py` | (Próximo) Update step |

---

### 4.1.3.1 Criterio de Convergencia (¡CLAVE!)

| Ecuación | Descripción | Módulo | Implementación |
|----------|-------------|--------|----------------|
| **4.59** | **Criterio de Gutiérrez:**<br/>‖**R**ᵉ(**U**_{n+1})‖ / ‖**F**ᵉˣᵗ_n‖ ≤ β | `convergence.py` | `GutierrezConvergence.check()` líneas ~95-120 |

**Parámetros típicos:**
- β = 0.001 (0.1% tolerancia relativa)
- Tolerancia absoluta: 1×10⁻⁶ N

**Por qué esta ecuación es clave:**
1. Evita auto-cancelación (ref ≠ residual)
2. Escala físicamente con nivel de carga
3. Funciona después de inserción de grietas
4. **Recomendación explícita de Gutiérrez para XFEM**

---

### 4.1.4 Transferencia de DOFs (Mapping Scheme)

| Ecuación | Descripción | Módulo | Implementación |
|----------|-------------|--------|----------------|
| **4.60** | Error de mapeo: e = ∫ (**U**ᵒˡᵈ - **U**ⁿᵉʷ)·(**U**ᵒˡᵈ - **U**ⁿᵉʷ) dA | `xfem_enrichment.py` | (Próximo) DOF transfer |
| **4.61** | **Sistema de mapeo:**<br/>**A**ⁿᵉʷ **U**ⁿᵉʷ = **A**ᵒˡᵈ **U**ᵒˡᵈ | `xfem_enrichment.py` | (Próximo) Transfer solver |
| **4.62-4.63** | Matrices de proyección | `xfem_enrichment.py` | (Próximo) Assembly |

---

## 🔧 Resumen por Módulo

### `cohesive_laws.py` (Implementado ✅)
**Ecuaciones:** 3.17, 3.18, 3.24-3.30, 4.52, 4.53

Funcionalidad:
- ✅ Ley de Reinhardt no-lineal
- ✅ Ley bilineal simplificada  
- ✅ Carga/descarga cíclica
- ✅ Rigidez tangente para Newton
- ✅ Tracción de corte con degradación

---

### `convergence.py` (Implementado ✅)
**Ecuación:** 4.59 (clave)

Funcionalidad:
- ✅ Criterio de Gutiérrez (force-scaled)
- ✅ Detección de estancamiento (opcional)
- ✅ Monitor de convergencia verboso
- ✅ Comparación con criterio naive

---

### `crack_criteria.py` (Implementado ✅)
**Ecuaciones:** 3.31-3.37

Funcionalidad:
- ✅ Esfuerzos principales (Rankine)
- ✅ Criterio de iniciación (σ_max ≥ f_t)
- ✅ Promediado no-local Gaussiano
- ✅ Criterio de propagación (σ̃_n ≥ f_t)
- ✅ Cálculo de ángulo de propagación

---

### `xfem_enrichment.py` (Próximo ⏳)
**Ecuaciones:** 4.60-4.63 (transferencia DOFs)

Funcionalidad pendiente:
- ⏳ Funciones de enriquecimiento (Heaviside, tip)
- ⏳ Construcción de matriz B enriquecida
- ⏳ Transferencia de DOFs entre topologías
- ⏳ Integración numérica en elementos partidos

---

### `newton_solver.py` (Próximo ⏳)
**Ecuaciones:** 4.48-4.51, 4.58

Funcionalidad pendiente:
- ⏳ Loop Newton-Raphson
- ⏳ Ensamble de residual y tangente
- ⏳ Line search (opcional)
- ⏳ Manejo de DOFs fijos

---

## 📊 Estadísticas de Implementación

| Categoría | Total | Implementadas | Pendientes |
|-----------|-------|---------------|------------|
| **Cohesive** | 11 | 11 ✅ | 0 |
| **Crack** | 7 | 7 ✅ | 0 |
| **Convergence** | 1 | 1 ✅ | 0 |
| **Newton** | 5 | 0 | 5 ⏳ |
| **XFEM** | 4 | 0 | 4 ⏳ |
| **TOTAL** | **28** | **19 (68%)** | **9 (32%)** |

---

## 🎯 Ecuaciones Más Importantes

### Top 5 Ecuaciones Críticas

1. **Eq. 4.59** - Criterio de Convergencia de Gutiérrez
   - **Por qué**: Soluciona el bug de convergencia del código original
   - **Dónde**: `convergence.py`, línea ~107
   - **Impacto**: Alto - previene subdivisión infinita

2. **Eq. 3.24** - Ley de Reinhardt
   - **Por qué**: Describe softening realista del concreto
   - **Dónde**: `cohesive_laws.py`, línea ~176
   - **Impacto**: Alto - física correcta de fractura

3. **Eq. 3.35-3.36** - Esfuerzo No-Local
   - **Por qué**: Independencia de malla en propagación
   - **Dónde**: `crack_criteria.py`, líneas ~139-168
   - **Impacto**: Medio - resultados mesh-objective

4. **Eq. 3.31** - Rankine (Iniciación)
   - **Por qué**: Determina cuándo/dónde inicia grieta
   - **Dónde**: `crack_criteria.py`, línea ~40
   - **Impacto**: Alto - evento crítico

5. **Eq. 4.53** - Rigidez Tangente Cohesiva
   - **Por qué**: Convergencia cuadrática de Newton
   - **Dónde**: `cohesive_laws.py`, línea ~184
   - **Impacto**: Alto - eficiencia numérica

---

## 📖 Referencias Cruzadas

### De Ecuación a Código
```python
# Buscar implementación de ecuación específica:

# Eq. 3.24 (Reinhardt)
from cohesive_laws import ReinhardtCohesiveLaw
law = ReinhardtCohesiveLaw(f_t=2.5e6, G_f=100.0)
# Ver líneas 176-186 para implementación exacta

# Eq. 4.59 (Convergencia)
from convergence import GutierrezConvergence
criterion = GutierrezConvergence(relative_beta=1e-3)
# Ver líneas 95-120 para algoritmo

# Eq. 3.35 (No-local)
from crack_criteria import NonLocalAveraging
averaging = NonLocalAveraging(influence_radius=0.08, length_scale=0.04)
# Ver líneas 139-168 para integración Gaussiana
```

### De Código a Ecuación
```python
# En el código, buscar comentarios con "Eq. X.YZ"
# Ejemplo en cohesive_laws.py:

def _traction_loading(self, omega: float) -> float:
    """Calculate normal traction during loading (Eq. 3.24)."""
    # ... implementación ...
```

---

## ✅ Validación

Cada ecuación implementada tiene:
1. ✅ Test unitario verificando valores conocidos
2. ✅ Docstring con referencia a ecuación
3. ✅ Comentario inline en código crítico
4. ✅ Ejemplo de uso en README

---

**Última actualización:** 2025-12-27  
**Completado:** 68% (19/28 ecuaciones)  
**Estado:** Módulos fundamentales implementados, solver completo pendiente
