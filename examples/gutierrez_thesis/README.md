# Gutiérrez Thesis Case Suite

Comprehensive implementation of all numerical examples from Chapter 5 of Gutiérrez's thesis on bond-slip in reinforced concrete with XFEM.

## Overview

This module implements **all** Chapter 5 examples:

1. **Pull-out tests** (Lettow): Bond-slip validation with empty elements
2. **SSPOT FRP**: FRP sheet debonding (bilinear bond law)
3. **Tensile members** (STN12): Distributed cracking with bond-slip
4. **3PB/4PB beams** (T5A1): Flexural cracking with reinforcement
5. **RC walls** (C1/C2): Cyclic loading with drift protocol
6. **Fibre-reinforced**: Banholzer bond law with random fibre generation

## Quick Start

```bash
# List all available cases
python -m examples.gutierrez_thesis.run --list

# Run a specific case
python -m examples.gutierrez_thesis.run --case pullout --mesh medium

# Run with overrides
python -m examples.gutierrez_thesis.run --case beam --nsteps 100 --mesh-factor 1.5
```

## Features Implemented

- ✅ All bond laws: CEB-FIP, Bilinear (FRP), Banholzer (fibres)
- ✅ Subdomain management (void elements, rigid regions)
- ✅ FRP sheet modeling
- ✅ Fibre generation with random orientation
- ✅ Cyclic loading with drift protocol
- ✅ Comprehensive postprocessing (slip profiles, bond stress, VTK export)
- ✅ CLI with flexible overrides
