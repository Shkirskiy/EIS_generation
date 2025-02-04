### Code Overview

- **Component Impedance Functions:**  
  `Z_component` returns the impedance for a given component type using standard formulas.

- **Circuit Parsing:**  
  The `parse_circuit` function (with helper functions) converts a circuit string into a tree structure, and `get_leaf_components` returns all component names.

- **Impedance Calculation:**  
  `compute_impedance` recursively computes the overall impedance of the circuit at a specified frequency.

- **Impedance Contribution Calculation:**  
  `compute_impedance_contributions` perturbs each leaf component by a factor (default 100% increase) to measure its absolute effect on the overall impedance, then normalizes these contributions.

- **Main Execution:**  
  The script parses a sample circuit, computes the overall impedance at a test frequency (0.001 Hz), prints the absolute impedance contribution percentages, and produces two plots:
  1. A Nyquist plot of the impedance spectrum.
  2. A semilog-x plot of impedance contribution percentages versus frequency.
