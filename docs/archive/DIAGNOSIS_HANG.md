# Diagnosis: Hang in run_analysis_xfem()

## Problem Statement
`validation_bond_slip_pullout.py` hangs indefinitely after printing the header.

## Root Cause Analysis

### Loop Structure
```python
# analysis_single.py:475-706
for istep in range(1, nsteps + 1):
    stack = [(u_n, du_total, 0)]
    while stack:  # Substepping loop
        u0, du, level = stack.pop()

        while True:  # Inner loop: Newton + crack updates
            ok, q_sol, ... = solve_step(...)

            if not ok:  # Newton failed
                need_split = True
                split_reason = "newton"
                break

            if not changed:  # No crack updates
                break

        if need_split:
            if level >= max_subdiv:
                raise RuntimeError("Substepping exceeded...")
            stack.append((u0 + 0.5 * du, 0.5 * du, level + 1))  # Subdivide
            stack.append((u0, 0.5 * du, level + 1))
            continue
```

### The Hang Scenario
From our test output:
```
[newton] stagnated      it=30 ||du||=9.825e-10 ||rhs||=5.220e+00 u=0.000mm
```

Newton is **stagnated** but not converged:
- `||du|| = 9.8e-10` < `newton_tol_du = 1e-9` → stagnation detected
- `||rhs|| = 5.22 N` >> `newton_tol_r = 1e-5` → NOT converged

In `solve_step()` (line 374-377):
```python
if norm_du < float(model.newton_tol_du):
    if model.debug_newton:
        print(f"[newton] stagnated ...")
    return False, q, ...  # Returns ok=False
```

This causes:
1. Newton fails → `ok = False`
2. Substepping triggered → `need_split = True`
3. Step subdivided → `stack.append((u0 + 0.5*du, ..., level+1))`
4. **Same problem at smaller step** → Newton fails again
5. Repeat until `level >= 12` → RuntimeError

### Why Newton Stagnates

The problem is **extreme ill-conditioning** despite diagonal scaling:

1. **At very small displacements** (u ≈ 24 nm):
   - Slip s ≈ 12 nm (half of applied displacement)
   - Bond stiffness k_bond ≈ 1e10 N/m (from test_bond_law_regularization.py)
   - Concrete stiffness K_concrete ≈ 1e9 N/m
   - **Ratio: k_bond / K_concrete ≈ 10×**

2. **Diagonal scaling can only help with diagonal dominance**, not singularity:
   - Scaling transforms D^(-1/2) K D^(-1/2)
   - Makes diagonal entries O(1)
   - **BUT** if the system is near-singular (rank-deficient), scaling can't fix it

3. **The bond-steel coupling creates a nearly singular system**:
   - Steel DOFs have very low stiffness (steel_EA_min = 1e3 N)
   - Bond DOFs have very high stiffness (k_bond ≈ 1e10 N/m at small slip)
   - This creates a huge stiffness contrast that leads to near-singularity

### Why It Hangs (Not Throws Error)

The code doesn't hang in an infinite loop, it's **slowly subdividing**:
- Each subdivision is slow because:
  1. Assembly with bond-slip + Numba compilation
  2. Newton iterations (30 iterations before stagnation)
  3. Diagonal scaling overhead

- With `nsteps=50` and `max_subdiv=12`:
  - First step subdivides 12 times before error
  - Each subdivision tries 30 Newton iterations
  - Total: 50 × 2^12 × 30 ≈ 61 million Newton iterations (worst case)
  - With buffered I/O, no output appears → looks like hang

## Solution Options

### Option 1: Fix the Pullout Test Setup (RECOMMENDED)
The pullout test is fundamentally different from a beam bending test:
- Should apply displacement to CONCRETE, not steel
- Steel displacement comes from bond-slip equilibrium
- Current setup applies BC directly to steel DOF → creates artificial constraint

**Implementation:**
```python
# In validation_bond_slip_pullout.py
# WRONG (current):
fixed = {load_dof: u_target}  # load_dof is steel DOF

# CORRECT:
# Apply displacement to concrete nodes at end
end_nodes = np.where(nodes[:, 0] > L - 1e-6)[0]
fixed = {2*n: u_target for n in end_nodes}
```

### Option 2: Increase s_reg Further
Current s_reg = 0.1 × s1 = 100 μm is still too small.

**Test shows:**
- At s = 100 nm: k_bond = 1.38e9 N/m (comparable to concrete!)
- At s = 1 μm: k_bond = 3.5e8 N/m (still high)

**Recommendation:**
```python
s_reg = 0.5 * s1  # 500 μm instead of 100 μm
```

### Option 3: Implement True Tangent Capping
Current tangent capping requires K_bulk which isn't available yet.

**Two-pass assembly:**
1. First pass: assemble K_bulk only (no bond-slip)
2. Compute dtau_max = bond_tangent_cap_factor × median(diag(K_bulk))
3. Second pass: assemble full system with capping

### Option 4: Detect Stagnation Loop
Add logic to detect repeated stagnation at same displacement:

```python
# In analysis_single.py
stagnation_count = 0
last_stagnation_u = None

if not ok and split_reason == "newton":
    if last_stagnation_u is not None and abs(u1 - last_stagnation_u) < 1e-12:
        stagnation_count += 1
        if stagnation_count > 3:
            raise RuntimeError(f"Repeated Newton stagnation at u={u1}")
    else:
        stagnation_count = 0
    last_stagnation_u = u1
```

## Immediate Fix for Testing

For now, to unblock testing:

1. **Disable bond-slip in pullout test** to verify rest of code works:
```python
model = XFEMModel(..., enable_bond_slip=False)
```

2. **Or reduce to tiny displacement**:
```python
u_max = 1e-9  # 1 nm (essentially zero)
nsteps = 1
```

3. **Or use a different test case** (beam bending) where BC setup is correct.

## Long-Term Solution

The fundamental issue is that **pullout tests require different BC setup** than beam tests:
- Pullout: Displacement applied to concrete, steel follows via bond
- Beam: Displacement applied to top, steel embedded in concrete

The current `run_analysis_xfem()` is designed for beam tests. A pullout-specific driver would be needed.
