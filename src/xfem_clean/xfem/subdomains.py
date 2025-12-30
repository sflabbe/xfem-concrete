"""
Element subdomains with property overrides.

Provides infrastructure for void elements, rigid regions, and bond masking.
This enables thesis cases with "empty elements" and "load elements".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np


@dataclass
class ElementProperties:
    """Properties for a single element.

    Attributes
    ----------
    material_type : str
        "bulk" (normal concrete), "void" (no stiffness), or "rigid" (very stiff)
    E_override : Optional[float]
        Override Young's modulus [Pa] (None = use material default)
    thickness_override : Optional[float]
        Override thickness [m] (None = use global thickness)
    bond_disabled : bool
        If True, disable bond-slip assembly for this element
    """

    material_type: str = "bulk"  # "bulk", "void", "rigid"
    E_override: Optional[float] = None
    thickness_override: Optional[float] = None
    bond_disabled: bool = False


class SubdomainManager:
    """
    Manages element-level property overrides for subdomains.

    Examples
    --------
    >>> # Create manager
    >>> mgr = SubdomainManager(nelem=100)
    >>>
    >>> # Mark void elements (x in [0, 20])
    >>> void_elems = [0, 1, 2, 3, 4]
    >>> for e in void_elems:
    ...     mgr.set_property(e, material_type="void", thickness_override=0.0)
    >>>
    >>> # Disable bond in empty element zone
    >>> for e in void_elems:
    ...     mgr.set_property(e, bond_disabled=True)
    >>>
    >>> # Query during assembly
    >>> props = mgr.get_property(2)
    >>> if props.material_type == "void":
    ...     # Skip bulk assembly
    ...     pass
    """

    def __init__(self, nelem: int):
        """
        Initialize subdomain manager.

        Parameters
        ----------
        nelem : int
            Total number of elements in mesh
        """
        self.nelem = nelem
        self._properties: Dict[int, ElementProperties] = {}

    def set_property(
        self,
        elem_id: int,
        material_type: Optional[str] = None,
        E_override: Optional[float] = None,
        thickness_override: Optional[float] = None,
        bond_disabled: Optional[bool] = None,
    ) -> None:
        """
        Set properties for an element.

        Parameters
        ----------
        elem_id : int
            Element index
        material_type : str, optional
            "bulk", "void", or "rigid"
        E_override : float, optional
            Override Young's modulus [Pa]
        thickness_override : float, optional
            Override thickness [m]
        bond_disabled : bool, optional
            Disable bond-slip for this element
        """
        if elem_id not in self._properties:
            self._properties[elem_id] = ElementProperties()

        props = self._properties[elem_id]

        if material_type is not None:
            if material_type not in ("bulk", "void", "rigid"):
                raise ValueError(f"Invalid material_type='{material_type}'. Use 'bulk', 'void', or 'rigid'.")
            props.material_type = material_type

        if E_override is not None:
            props.E_override = float(E_override)

        if thickness_override is not None:
            props.thickness_override = float(thickness_override)

        if bond_disabled is not None:
            props.bond_disabled = bool(bond_disabled)

    def get_property(self, elem_id: int) -> ElementProperties:
        """
        Get properties for an element.

        Parameters
        ----------
        elem_id : int
            Element index

        Returns
        -------
        props : ElementProperties
            Element properties (default if not overridden)
        """
        return self._properties.get(elem_id, ElementProperties())

    def is_void(self, elem_id: int) -> bool:
        """Check if element is void (no stiffness)."""
        return self.get_property(elem_id).material_type == "void"

    def is_rigid(self, elem_id: int) -> bool:
        """Check if element is rigid (very stiff)."""
        return self.get_property(elem_id).material_type == "rigid"

    def is_bond_disabled(self, elem_id: int) -> bool:
        """Check if bond-slip is disabled for element."""
        return self.get_property(elem_id).bond_disabled

    def get_effective_E(self, elem_id: int, E_default: float) -> float:
        """
        Get effective Young's modulus for element.

        Parameters
        ----------
        elem_id : int
            Element index
        E_default : float
            Default Young's modulus [Pa]

        Returns
        -------
        E : float
            Effective Young's modulus [Pa]
        """
        props = self.get_property(elem_id)

        # Override if specified
        if props.E_override is not None:
            return props.E_override

        # Void elements: E ≈ 0 (use very small value to avoid singularity)
        if props.material_type == "void":
            return 1e-6 * E_default  # Essentially zero

        # Rigid elements: E >> E_default
        if props.material_type == "rigid":
            return 1e6 * E_default  # Very stiff

        # Normal elements: use default
        return E_default

    def get_effective_thickness(self, elem_id: int, b_default: float) -> float:
        """
        Get effective thickness for element.

        Parameters
        ----------
        elem_id : int
            Element index
        b_default : float
            Default thickness [m]

        Returns
        -------
        b : float
            Effective thickness [m]
        """
        props = self.get_property(elem_id)

        if props.thickness_override is not None:
            return props.thickness_override

        return b_default


def build_subdomain_manager_from_config(
    nodes: np.ndarray,
    elems: np.ndarray,
    subdomains: List,
    unit_conversion: float = 1e-3,  # Default: mm → m
) -> SubdomainManager:
    """
    Build SubdomainManager from case configuration.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates [nnode, 2] in SI units (m)
    elems : np.ndarray
        Element connectivity [nelem, nnodes_per_elem]
    subdomains : List[SubdomainConfig]
        Subdomain configurations from case
    unit_conversion : float
        Conversion factor from config units to node units (default: 1e-3 for mm→m)

    Returns
    -------
    mgr : SubdomainManager
        Configured subdomain manager
    """
    nelem = elems.shape[0]
    mgr = SubdomainManager(nelem)

    # Process each subdomain configuration
    for subdomain in subdomains:
        # Find elements in subdomain
        elem_ids = []

        if subdomain.element_indices is not None:
            # Explicit element list
            elem_ids = subdomain.element_indices

        elif subdomain.x_range is not None or subdomain.y_range is not None:
            # Spatial region (convert from config units to SI units)
            if subdomain.x_range is not None:
                x_min = subdomain.x_range[0] * unit_conversion
                x_max = subdomain.x_range[1] * unit_conversion
            else:
                x_min, x_max = -1e30, 1e30

            if subdomain.y_range is not None:
                y_min = subdomain.y_range[0] * unit_conversion
                y_max = subdomain.y_range[1] * unit_conversion
            else:
                y_min, y_max = -1e30, 1e30

            for e in range(nelem):
                # Get element centroid (already in SI units)
                elem_nodes = nodes[elems[e], :]
                xc = np.mean(elem_nodes[:, 0])
                yc = np.mean(elem_nodes[:, 1])

                # Check if centroid in range
                if x_min <= xc <= x_max and y_min <= yc <= y_max:
                    elem_ids.append(e)

        # Apply overrides to selected elements
        for e in elem_ids:
            mgr.set_property(
                e,
                material_type=subdomain.material_type,
                E_override=subdomain.E_override,
                thickness_override=subdomain.thickness_override,
                bond_disabled=(subdomain.material_type == "void"),  # Auto-disable bond in void
            )

    return mgr


def get_bond_disabled_segments(
    rebar_segs: np.ndarray,
    nodes: np.ndarray,
    bond_disabled_x_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Get mask of bond-disabled segments based on spatial range.

    Parameters
    ----------
    rebar_segs : np.ndarray
        Rebar segments [n_seg, 5]: [n1, n2, L0, cx, cy]
    nodes : np.ndarray
        Node coordinates [nnode, 2]
    bond_disabled_x_range : Tuple[float, float], optional
        (x_min, x_max) range where bond is disabled

    Returns
    -------
    mask : np.ndarray
        Boolean mask [n_seg] where True = bond disabled
    """
    n_seg = rebar_segs.shape[0]
    mask = np.zeros(n_seg, dtype=bool)

    if bond_disabled_x_range is None:
        return mask

    x_min, x_max = bond_disabled_x_range

    for i in range(n_seg):
        n1 = int(rebar_segs[i, 0])
        n2 = int(rebar_segs[i, 1])

        # Get segment midpoint x-coordinate
        x_mid = 0.5 * (nodes[n1, 0] + nodes[n2, 0])

        # Disable if midpoint in range
        if x_min <= x_mid <= x_max:
            mask[i] = True

    return mask
