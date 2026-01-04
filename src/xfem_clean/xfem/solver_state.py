"""Unified solver state container for XFEM analysis."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Union, List, Any

from xfem_clean.bond_slip import BondSlipStateArrays
from xfem_clean.xfem.state_arrays import (
    BulkStateArrays,
    BulkStatePatch,
    CohesiveStateArrays,
    CohesiveStatePatch,
)


@dataclass
class SolverState:
    """Container for all persistent state variables in the solver.

    Attributes
    ----------
    mp : BulkStateArrays
        Material point states (stress, strain, damage, plasticity).
    coh : CohesiveStateArrays
        Cohesive interface states (max opening, damage).
    bond : Optional[Union[BondSlipStateArrays, List[BondSlipStateArrays]]]
        Bond-slip states (slip, shear stress). Can be a single array (legacy)
        or a list of arrays (multi-layer).
    """

    mp: BulkStateArrays
    coh: CohesiveStateArrays
    bond: Optional[Union[BondSlipStateArrays, List[BondSlipStateArrays]]] = None

    def copy(self) -> "SolverState":
        """Create a deep copy of the state."""
        # BulkStateArrays and CohesiveStateArrays have specific copy methods
        new_mp = self.mp.copy()
        new_coh = self.coh.copy()
        
        # Bond state handling
        new_bond = None
        if self.bond is not None:
            if isinstance(self.bond, list):
                new_bond = [b.copy() for b in self.bond]
            else:
                new_bond = self.bond.copy()

        return SolverState(mp=new_mp, coh=new_coh, bond=new_bond)

    def update(
        self,
        mp_patch: Optional[BulkStatePatch] = None,
        coh_patch: Optional[CohesiveStatePatch] = None,
        bond_new: Optional[Union[BondSlipStateArrays, List[BondSlipStateArrays]]] = None,
    ) -> None:
        """Apply updates to the state in-place.

        Parameters
        ----------
        mp_patch : Optional[BulkStatePatch]
            Sparse updates for material points.
        coh_patch : Optional[CohesiveStatePatch]
            Sparse updates for cohesive interfaces.
        bond_new : Optional[Union[BondSlipStateArrays, List[BondSlipStateArrays]]]
            New full state for bond-slip (since bond assembly typically returns full new arrays).
        """
        if mp_patch is not None:
            mp_patch.apply_to(self.mp)
        
        if coh_patch is not None:
            coh_patch.apply_to(self.coh)
            
        if bond_new is not None:
            # Bond slip assembly returns fresh copies currently, so we replace
            # We could optimize this to patch-based later if needed
            self.bond = bond_new
