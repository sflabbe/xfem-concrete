"""Benchmark bond-slip assembly performance (Numba vs Python).

This script measures the performance improvement from Numba-accelerated
bond-slip assembly compared to the pure Python fallback.

Usage:
    python -m benchmarks.benchmark_bond_slip
    python -m benchmarks.benchmark_bond_slip --nseg 1000,5000,10000 --repeat 5
    python -m benchmarks.benchmark_bond_slip --output results.csv

Expected Results:
    - For n_seg >= 1000: Numba should be 2-10× faster than Python
    - Speedup increases with problem size (better amortization of JIT overhead)
    - First run includes JIT compilation time; subsequent runs are cached
"""

import argparse
import time
import csv
from pathlib import Path

import numpy as np

from xfem_clean.bond_slip import (
    assemble_bond_slip,
    BondSlipModelCode2010,
    BondSlipStateArrays,
)


def generate_synthetic_problem(n_seg: int, ndof_total: int):
    """Generate synthetic bond-slip problem for benchmarking.

    Parameters
    ----------
    n_seg : int
        Number of bond-slip segments
    ndof_total : int
        Total number of DOFs (concrete + steel)

    Returns
    -------
    u_total : np.ndarray
        Displacement vector [ndof_total]
    steel_segments : np.ndarray
        Segment geometry [n_seg, 5]
    steel_dof_map : np.ndarray
        DOF mapping [nnode, 2]
    bond_law : BondSlipModelCode2010
        Bond-slip constitutive model
    bond_states : BondSlipStateArrays
        Bond-slip states
    """
    # Generate segments: uniformly distributed along x-axis
    nnode = n_seg + 1
    x = np.linspace(0, 1.0, nnode)

    steel_segments = np.zeros((n_seg, 5), dtype=float)
    for i in range(n_seg):
        n1, n2 = i, i + 1
        dx = x[n2] - x[n1]
        dy = 0.0
        L0 = np.sqrt(dx**2 + dy**2)
        cx = dx / L0
        cy = dy / L0
        steel_segments[i] = [n1, n2, L0, cx, cy]

    # DOF mapping: concrete DOFs 0..2*nnode-1, steel DOFs 2*nnode..4*nnode-1
    concrete_dofs = 2 * nnode
    steel_dof_offset = concrete_dofs
    steel_dof_map = -np.ones((nnode, 2), dtype=np.int64)
    for n in range(nnode):
        steel_dof_map[n, 0] = steel_dof_offset + 2 * n
        steel_dof_map[n, 1] = steel_dof_offset + 2 * n + 1

    # Displacement vector: random small displacements
    u_total = np.random.randn(ndof_total) * 1e-5  # meters

    # Bond law
    bond_law = BondSlipModelCode2010(
        d_bar=0.016,  # 16mm rebar
        f_cm=30e6,    # 30 MPa concrete
        condition="good",
    )

    # Bond states
    bond_states = BondSlipStateArrays.zeros(n_segments=n_seg)

    return u_total, steel_segments, steel_dof_map, bond_law, bond_states, steel_dof_offset


def benchmark_bond_slip(n_seg: int, use_numba: bool, n_repeat: int = 3):
    """Benchmark bond-slip assembly for given problem size.

    Parameters
    ----------
    n_seg : int
        Number of bond-slip segments
    use_numba : bool
        Use Numba acceleration
    n_repeat : int
        Number of times to repeat (use min time)

    Returns
    -------
    time_sec : float
        Minimum execution time (seconds)
    """
    # Generate problem
    nnode = n_seg + 1
    concrete_dofs = 2 * nnode
    steel_dofs = 2 * nnode
    ndof_total = concrete_dofs + steel_dofs

    u_total, steel_segments, steel_dof_map, bond_law, bond_states, steel_dof_offset = \
        generate_synthetic_problem(n_seg, ndof_total)

    # Warm-up run (compile Numba kernel if needed)
    _ = assemble_bond_slip(
        u_total=u_total,
        steel_segments=steel_segments,
        steel_dof_offset=steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=steel_dof_map,
        steel_EA=200e9 * (np.pi * 0.008**2),
        use_numba=use_numba,
        perimeter=np.pi * 0.016,
        bond_gamma=1.0,
        enable_validation=False,  # Skip validation for speed
    )

    # Benchmark runs
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        f_bond, K_bond, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=200e9 * (np.pi * 0.008**2),
            use_numba=use_numba,
            perimeter=np.pi * 0.016,
            bond_gamma=1.0,
            enable_validation=False,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark bond-slip assembly (Numba vs Python)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--nseg",
        type=str,
        default="100,500,1000,5000,10000",
        help="Comma-separated segment counts (default: 100,500,1000,5000,10000)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repetitions (use min time) (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: print to stdout)"
    )

    args = parser.parse_args()

    # Parse segment counts
    n_segs = [int(x.strip()) for x in args.nseg.split(",")]

    print("Bond-Slip Assembly Benchmark")
    print("=" * 70)
    print(f"Segment counts: {n_segs}")
    print(f"Repetitions: {args.repeat}")
    print()

    # Collect results
    results = []

    for n_seg in n_segs:
        print(f"Benchmarking n_seg={n_seg}...")

        # Python fallback
        time_python = benchmark_bond_slip(n_seg, use_numba=False, n_repeat=args.repeat)

        # Numba kernel
        time_numba = benchmark_bond_slip(n_seg, use_numba=True, n_repeat=args.repeat)

        speedup = time_python / time_numba if time_numba > 0 else float('inf')

        results.append({
            'n_seg': n_seg,
            'time_python_sec': time_python,
            'time_numba_sec': time_numba,
            'speedup': speedup,
        })

        print(f"  Python: {time_python:.6f} s")
        print(f"  Numba:  {time_numba:.6f} s")
        print(f"  Speedup: {speedup:.2f}×")
        print()

    # Save or print results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['n_seg', 'time_python_sec', 'time_numba_sec', 'speedup'])
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to: {output_path}")
    else:
        print("\nResults Summary:")
        print("-" * 70)
        print(f"{'n_seg':>8} {'Python (s)':>12} {'Numba (s)':>12} {'Speedup':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['n_seg']:8d} {r['time_python_sec']:12.6f} {r['time_numba_sec']:12.6f} {r['speedup']:10.2f}×")

    # Performance summary
    print("\nPerformance Summary:")
    print("-" * 70)
    avg_speedup = np.mean([r['speedup'] for r in results if r['n_seg'] >= 1000])
    print(f"Average speedup (n_seg >= 1000): {avg_speedup:.2f}×")

    if avg_speedup >= 2.0:
        print("✅ Performance target met (>= 2.0× for large problems)")
    else:
        print(f"⚠️  Performance below target (expected >= 2.0×, got {avg_speedup:.2f}×)")


if __name__ == "__main__":
    main()
