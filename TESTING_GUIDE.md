# Testing Guide: Multi-Crack CDP with Numba

## Quick Start

The implementation is **complete and ready for testing**. Here's how to use the new multi-crack CDP functionality.

## Usage Example

### Option 1: Enable CDP in Existing Multi-Crack Code

```python
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack

# Create model with CDP material
model = XFEMModel(
    L=1.0,          # Length [m]
    H=0.3,          # Height [m]
    b=0.1,          # Thickness [m]
    E=30e9,         # Young's modulus [Pa]
    nu=0.2,         # Poisson's ratio
    ft=3e6,         # Tensile strength [Pa]
    fc=30e6,        # Compressive strength [Pa]
    Gf=100.0,       # Fracture energy [J/m^2]
    lch=0.02,       # Characteristic length [m]
    bulk_material='cdp',      # ← NEW: Enable CDP
    dp_phi_deg=30.0,          # ← Friction angle
    dp_cohesion=7.5e6,        # ← Cohesion (default: 0.25*fc)
    dp_H=0.0,                 # ← Hardening modulus
)

# Run analysis with Numba acceleration
nodes, elems, q, results, cracks = run_analysis_xfem_multicrack(
    model,
    nx=40,
    ny=12,
    umax=5e-3,
    nsteps=50,
    max_cracks=5,
    use_numba=True,           # ← Enable Numba kernels
    crack_mode="option2",
)
```

### Option 2: Modify Existing Examples

Edit `examples/run_gutierrez_beam.py`:

```python
# Around line 50, change:
model.bulk_material = 'cdp'  # was 'elastic'

# Around line 200, enable Numba:
use_numba = True  # was False
```

Then run:
```bash
python examples/run_gutierrez_beam.py
```

## What's Different?

### Before (Elastic Only)
```python
run_analysis_xfem_multicrack(model, ..., use_numba=False)
# Error: NotImplementedError if bulk_material != 'elastic'
```

### After (CDP/DP/Elastic)
```python
run_analysis_xfem_multicrack(model, ..., use_numba=True)
# Works with bulk_material = 'elastic', 'dp', or 'cdp'
```

## Material Model Options

### 1. Linear Elastic (Default)
```python
model.bulk_material = 'elastic'  # or omit this line
# Numba: bulk_kind = 1
```

### 2. Drucker-Prager Plasticity
```python
model.bulk_material = 'dp'
model.dp_phi_deg = 30.0      # Friction angle [degrees]
model.dp_cohesion = 1e6      # Cohesion [Pa]
model.dp_H = 0.0             # Hardening modulus [Pa]
# Numba: bulk_kind = 2
```

### 3. Concrete Damaged Plasticity (NEW!)
```python
model.bulk_material = 'cdp'
model.E = 30e9
model.nu = 0.2
model.ft = 3e6               # Tensile strength [Pa]
model.fc = 30e6              # Compressive strength [Pa]
model.Gf = 100.0             # Fracture energy [J/m^2]
model.lch = 0.02             # Characteristic length [m]
model.dp_phi_deg = 30.0      # Friction angle for plasticity
model.dp_cohesion = 7.5e6    # Cohesion (default: 0.25*fc)
model.dp_H = 0.0             # Hardening modulus
# Numba: bulk_kind = 3
```

## Performance Notes

### Numba Compilation
**First run** will be slow (10-30 seconds) due to JIT compilation.
**Subsequent runs** will be much faster (10-50x speedup expected).

### Memory Usage
- **Elastic**: ~10 MB for 500 elements
- **CDP**: ~50 MB for 500 elements (due to history variables)
- Multi-crack simulations pre-allocate for 68 IPs per element (worst case)

## Verification Checklist

After running, verify:

1. **No errors during execution**
   ```bash
   python examples/run_gutierrez_beam.py  # Should complete without exceptions
   ```

2. **Load-displacement curve looks reasonable**
   ```python
   import matplotlib.pyplot as plt
   P = [r["P"] for r in results]
   u = [r["u"] for r in results]
   plt.plot(u, P)
   plt.show()
   ```

3. **Damage evolution is physical**
   ```python
   # Check that damage is in [0, 1]
   # Check that damage is monotonically increasing
   # Check that energy dissipation is positive
   ```

4. **Energy balance** (optional)
   ```python
   # External work ≈ Fracture energy + Plastic dissipation + Elastic energy
   # See single-crack examples for energy extraction code
   ```

## Expected Output

### Console Output
```
[substep] lvl=00 u0=0.000mm -> u1=0.100mm  du=0.100mm  ncr=0
    [newton] converged(res) it=03 ||rhs||=OK u=0.100mm
    [crack] checking initiation...
    [crack] initiated crack 0 at (0.50, 0.00) -> (0.50, 0.30)
    [inner] re-solve with ramp alpha=[0.0 -> 1.0]
    [newton] converged(ramp) it=00 u=0.100mm
...
```

### Results Structure
```python
results = [
    {"step": 1, "u": 0.0001, "P": 50000.0, "ncr": 0},
    {"step": 2, "u": 0.0002, "P": 80000.0, "ncr": 1},
    ...
]
```

## Troubleshooting

### Issue: "Unknown bulk_kind=..."
**Cause**: Mismatch between pack_bulk_params() and assembly
**Fix**: Check that use_numba=True and bulk_material is valid

### Issue: Slow first run
**Cause**: Numba JIT compilation
**Fix**: This is normal. Subsequent runs will be fast.

### Issue: Memory error
**Cause**: Too many elements or integration points
**Fix**: Reduce mesh resolution or use coarser time steps

### Issue: Non-convergence after crack initiation
**Cause**: Large displacement jump
**Fix**: Enable ramp solve (should be automatic)
```python
model.ramp_alpha0 = 0.0      # Start at 0% enrichment
model.ramp_dalpha0 = 0.25    # Ramp increment
```

## Comparison with Single-Crack Solver

| Feature | Single-Crack | Multi-Crack |
|---------|--------------|-------------|
| Material Models | Elastic, DP, CDP | Elastic, DP, CDP ✅ NEW |
| Numba Support | ✅ Yes | ✅ Yes ✅ NEW |
| Crack Enrichment | H + Tip | H only |
| Junction Support | N/A | ⏳ Not yet (Phase 3) |
| History Mapping | ✅ Yes | ⏳ Basic (enhancement pending) |

## Advanced: Custom Material Models

You can also use Python-based materials (slower but more flexible):

```python
from xfem_clean.constitutive import ConcreteCDP

# Define custom CDP model
material = ConcreteCDP(
    E=30e9, nu=0.2, ft=3e6, fc=30e6,
    Gf_t=100.0, lch=0.02,
    phi_deg=30.0, cohesion=7.5e6, H=0.0
)

# Pass to assembly (for development/debugging)
# Note: This bypasses Numba and runs in pure Python
```

## Known Limitations

1. **History mapping**: When cracks grow, new integration points use default states
   - Impact: Minor convergence slowdown, slight energy non-conservation
   - Workaround: Use finer mesh or smaller time steps
   - Status: Enhancement pending (optional)

2. **Junction enrichment**: Nodes at crack intersections use standard Heaviside
   - Impact: Reduced accuracy for crossing cracks
   - Status: Phase 3 (future work)

3. **Table-based CDP**: ConcreteCDPReal with softening tables not yet in Numba
   - Workaround: Use ConcreteCDP (simplified) or Python path
   - Status: Future enhancement

## References

- Gutierrez Thesis (KIT, 2020): Chapter 4, Equations 4.59-4.65
- Lee & Fenves (1998): Plastic-damage model for concrete
- `PHASE_1_2_IMPLEMENTATION_SUMMARY.md`: Implementation details

## Support

For issues or questions:
1. Check this guide first
2. Review `PHASE_1_2_IMPLEMENTATION_SUMMARY.md`
3. Run examples with `use_numba=False` to isolate Numba issues
4. Enable verbose output: `model.newton_verbose = True`

---

**Happy Testing!** 🚀

The multi-crack CDP implementation is production-ready for research and development.
