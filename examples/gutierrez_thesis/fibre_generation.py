"""
Fibre Generation and Management for FRC

Generates random fibre distributions in crack zones and manages fibre-crack coupling.

Reference: Banholzer et al., Gutiérrez thesis Chapter 5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math


@dataclass
class Fibre:
    """Single fibre instance"""
    # Geometry
    center_x: float  # mm
    center_y: float  # mm
    length: float  # mm
    diameter: float  # mm
    orientation: float  # radians (0 = horizontal)

    # State
    is_bridging: bool = False  # True if fibre crosses crack
    crack_intersection: Optional[Tuple[float, float]] = None  # (x, y) of intersection
    embedment_length_1: float = 0.0  # Length on side 1 of crack
    embedment_length_2: float = 0.0  # Length on side 2 of crack

    # Forces
    bond_force: float = 0.0  # N (total force from bond-slip)
    slip: float = 0.0  # m (current slip)
    slip_max: float = 0.0  # m (historical maximum)

    def endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get fibre endpoints"""
        dx = 0.5 * self.length * math.cos(self.orientation)
        dy = 0.5 * self.length * math.sin(self.orientation)
        p1 = (self.center_x - dx, self.center_y - dy)
        p2 = (self.center_x + dx, self.center_y + dy)
        return p1, p2


class FibrePopulation:
    """
    Manages a population of discrete fibres in FRC.

    Parameters
    ----------
    fibres : List[Fibre]
        List of fibre instances
    bond_law : BanholzerBondLaw
        Bond-slip constitutive law for fibres
    """

    def __init__(self, fibres: List[Fibre], bond_law: Any):
        self.fibres = fibres
        self.bond_law = bond_law
        self.n_fibres = len(fibres)
        self.n_bridging = 0

    def update_crack_intersections(self, crack_x: float, crack_y_range: Tuple[float, float]):
        """
        Update which fibres are bridging the crack.

        Parameters
        ----------
        crack_x : float
            x-coordinate of vertical crack (mm)
        crack_y_range : Tuple[float, float]
            (y_min, y_max) range of crack (mm)
        """
        self.n_bridging = 0
        y_min, y_max = crack_y_range

        for fib in self.fibres:
            p1, p2 = fib.endpoints()
            x1, y1 = p1
            x2, y2 = p2

            # Check if fibre crosses crack line (simplified: vertical crack)
            if (x1 <= crack_x <= x2) or (x2 <= crack_x <= x1):
                # Compute intersection y-coordinate
                if abs(x2 - x1) < 1e-9:
                    # Vertical fibre - no intersection with vertical crack
                    fib.is_bridging = False
                    continue

                t = (crack_x - x1) / (x2 - x1)
                y_intersect = y1 + t * (y2 - y1)

                # Check if intersection is within crack extent
                if y_min <= y_intersect <= y_max:
                    fib.is_bridging = True
                    fib.crack_intersection = (crack_x, y_intersect)

                    # Compute embedment lengths
                    dx1 = crack_x - x1
                    dy1 = y_intersect - y1
                    fib.embedment_length_1 = math.sqrt(dx1**2 + dy1**2)

                    dx2 = x2 - crack_x
                    dy2 = y2 - y_intersect
                    fib.embedment_length_2 = math.sqrt(dx2**2 + dy2**2)

                    self.n_bridging += 1
                else:
                    fib.is_bridging = False
            else:
                fib.is_bridging = False

    def compute_bridging_forces(self, crack_opening: np.ndarray):
        """
        Compute bridging forces for all active fibres.

        Parameters
        ----------
        crack_opening : np.ndarray
            [n_gp] crack opening at Gauss points along crack (m)

        Returns
        -------
        total_bridging_stress : float
            Total bridging stress σ_fib (Pa) averaged over crack area
        """
        total_force = 0.0  # N

        for fib in self.fibres:
            if not fib.is_bridging:
                continue

            # Simplified: assume crack opening is uniform or sampled at fibre location
            # For full implementation, interpolate crack_opening to fibre intersection
            w = np.mean(crack_opening) if len(crack_opening) > 0 else 0.0

            # Slip is projection of crack opening onto fibre axis
            # Simplified: slip ≈ crack opening for small angles
            fib.slip = abs(w * math.cos(fib.orientation))
            fib.slip_max = max(fib.slip_max, fib.slip)

            # Bond stress from Banholzer law
            tau, dtau_ds = self.bond_law.tau_and_tangent(fib.slip, fib.slip_max)

            # Effective embedment length (shorter of the two sides)
            L_emb = min(fib.embedment_length_1, fib.embedment_length_2) * 1e-3  # mm → m

            # Bond force per fibre (assuming full circumference bonded)
            perimeter = math.pi * fib.diameter * 1e-3  # mm → m
            fib.bond_force = tau * perimeter * L_emb

            total_force += fib.bond_force

        # Convert to stress (force per unit area)
        # Assume fibres are distributed over a representative area
        # TODO: This should be computed from fibre density and specimen geometry
        return total_force

    def get_bridging_fibres(self) -> List[Fibre]:
        """Return list of currently bridging fibres"""
        return [f for f in self.fibres if f.is_bridging]


def generate_fibres_in_band(
    center_x: float,
    center_y: float,
    band_width: float,
    band_height: float,
    fibre_length: float,
    fibre_diameter: float,
    density: float,  # fibres per cm²
    random_seed: int = 42,
    orientation_distribution: str = "uniform",  # "uniform", "aligned"
    mean_orientation: float = 0.0,  # radians
) -> List[Fibre]:
    """
    Generate random fibre distribution in a band (crack zone).

    Parameters
    ----------
    center_x, center_y : float
        Center of band (mm)
    band_width, band_height : float
        Band dimensions (mm)
    fibre_length, fibre_diameter : float
        Fibre geometry (mm)
    density : float
        Fibre density (fibres per cm²)
    random_seed : int
        Random seed for reproducibility
    orientation_distribution : str
        "uniform" (0 to π) or "aligned" (around mean_orientation)
    mean_orientation : float
        Mean orientation for aligned distribution (radians, 0 = horizontal)

    Returns
    -------
    fibres : List[Fibre]
        List of generated fibres
    """
    rng = np.random.RandomState(random_seed)

    # Compute number of fibres
    area_cm2 = (band_width * band_height) / 100.0  # mm² → cm²
    n_fibres = int(density * area_cm2)

    fibres = []

    for i in range(n_fibres):
        # Random position in band
        x = center_x + (rng.rand() - 0.5) * band_width
        y = center_y + (rng.rand() - 0.5) * band_height

        # Random orientation
        if orientation_distribution == "uniform":
            theta = rng.rand() * math.pi  # 0 to π
        elif orientation_distribution == "aligned":
            # Normal distribution around mean
            theta = mean_orientation + rng.randn() * 0.1  # ±0.1 rad std dev
        else:
            theta = 0.0

        fib = Fibre(
            center_x=x,
            center_y=y,
            length=fibre_length,
            diameter=fibre_diameter,
            orientation=theta,
        )

        fibres.append(fib)

    return fibres


def generate_fibres_for_case(
    case_config,  # CaseConfig with fibres defined
    crack_expected_x: float,  # mm
    crack_expected_y_range: Tuple[float, float],  # mm
) -> FibrePopulation:
    """
    Generate fibre population for a thesis case.

    Parameters
    ----------
    case_config : CaseConfig
        Case configuration with fibres defined
    crack_expected_x : float
        Expected x-coordinate of crack (mm)
    crack_expected_y_range : Tuple[float, float]
        Expected (y_min, y_max) of crack (mm)

    Returns
    -------
    population : FibrePopulation
        Generated fibre population with bond law
    """
    if case_config.fibres is None:
        return FibrePopulation([], None)

    fibre_config = case_config.fibres.fibre
    bond_config = case_config.fibres.bond_law

    # Import bond law mapper
    from examples.gutierrez_thesis.solver_interface import map_bond_law
    bond_law = map_bond_law(bond_config)

    # Generate fibres in crack band
    y_min, y_max = crack_expected_y_range
    band_center_y = 0.5 * (y_min + y_max)
    band_height = (y_max - y_min) if case_config.fibres.active_near_crack_only else case_config.geometry.height

    # Band width: ±activation_distance around crack
    band_width = 2.0 * case_config.fibres.activation_distance

    fibres = generate_fibres_in_band(
        center_x=crack_expected_x,
        center_y=band_center_y,
        band_width=band_width,
        band_height=band_height,
        fibre_length=fibre_config.length,
        fibre_diameter=fibre_config.diameter,
        density=fibre_config.density,
        random_seed=case_config.fibres.random_seed,
        orientation_distribution="uniform",
        mean_orientation=fibre_config.orientation_deg * math.pi / 180.0,
    )

    population = FibrePopulation(fibres, bond_law)

    print(f"Generated {len(fibres)} fibres in crack band")
    print(f"  Band: x={crack_expected_x}±{band_width/2:.1f} mm, y={y_min:.1f}-{y_max:.1f} mm")
    print(f"  Fibre length: {fibre_config.length} mm, diameter: {fibre_config.diameter} mm")
    print(f"  Density: {fibre_config.density} fibres/cm²")

    return population
