#!/usr/bin/env python3
"""
Impedance Contribution Calculator for EIS Circuits

This script parses an equivalent circuit defined by a string and computes the overall
impedance using standard formulas for resistors, capacitors, inductors, Warburg elements,
and constant phase elements (CPE). It then estimates the absolute contribution of each
leaf component to the overall impedance using a finite perturbation method (doubling each
component value). Finally, it generates:
  - A Nyquist plot (Real(Z) vs -Imag(Z)) of the impedance spectrum.
  - A plot showing the absolute impedance contribution percentage of each component
    versus frequency.

Example circuit (string format):
    'R0-p(R1,C1)-p(R2,C2)-Wo1'

Usage:
  - Adjust the circuit string and component parameters as desired.
  - Run the script to see printed impedance values and generated plots.
  
Outputs:
  - Printed overall impedance at a test frequency.
  - Printed absolute impedance contribution percentages at the test frequency.
  - A Nyquist plot of the impedance spectrum over a user-specified frequency range.
  - A semilog-x plot showing the impedance contribution percentages of each leaf
    component versus frequency.

Author: Slava SHKIRSKIY
Date: 04/02/2025
"""

import numpy as np
import re
import copy
import matplotlib.pyplot as plt

# =============================================================
# Component Impedance Functions
# =============================================================
def Z_component(comp, freq, params):
    """
    Compute the complex impedance of a single circuit component at a given frequency.
    
    Supported component types:
      - Resistor ('R...'):        Z = R
      - Capacitor ('C...'):        Z = 1/(jωC)
      - Inductor ('L...'):         Z = jωL
      - Warburg element ('Wo...'): Z = sigma/√ω * exp(-jπ/4)
      - Constant Phase Element ('CPE...'):
           Z = 1 / (Q * (jω)**alpha)
         (For CPE, the magnitude Q is taken from key 'CPEX' and the exponent
          from key 'CPE_alphaX', where X is an optional numeric index.)
    
    Parameters:
      comp   : str, component name (e.g., 'R0', 'C1', 'Wo1')
      freq   : float, frequency in Hz
      params : dict, mapping component names to parameter values
    
    Returns:
      Complex impedance of the component.
    """
    omega = 2 * np.pi * freq
    if comp.startswith('R'):
        return params[comp]
    elif comp.startswith('C'):
        C_val = params[comp]
        if C_val == 0:
            return np.inf
        return 1 / (1j * omega * C_val)
    elif comp.startswith('L'):
        return 1j * omega * params[comp]
    elif comp.startswith('Wo'):
        sigma = params[comp]
        return sigma / np.sqrt(omega) * np.exp(-1j * np.pi / 4)
    elif comp.startswith('CPE'):
        Q = params.get(comp, None)
        m = re.match(r'CPE(\d+)', comp)
        if m:
            alpha_key = 'CPE_alpha' + m.group(1)
        else:
            alpha_key = 'CPE_alpha'
        alpha = params.get(alpha_key, None)
        if Q is None or alpha is None:
            raise ValueError("Missing parameter for CPE element " + comp)
        return 1 / (Q * (1j * omega)**alpha)
    else:
        raise ValueError("Unknown component type: " + comp)

# =============================================================
# Circuit Parsing Functions
# =============================================================
#
# The circuit string is assumed to follow a simple grammar:
#
#   circuit  := series
#   series   := term ('-' term)*
#   term     := component | parallel
#   parallel := 'p(' series (',' series)* ')'
#
# The parser converts the string into a tree structure of Node objects.
#

class Node:
    def __init__(self, node_type, children=None, value=None):
        """
        Initialize a Node.
        
        node_type: 'series', 'parallel', or 'component'
        children : list of child Node objects (if any)
        value    : for a component node, the component's name (string)
        """
        self.node_type = node_type
        self.children = children if children is not None else []
        self.value = value

    def __repr__(self):
        if self.node_type == 'component':
            return self.value
        else:
            sep = '-' if self.node_type == 'series' else '|'
            return "(" + sep.join(repr(child) for child in self.children) + ")"

def parse_circuit(s):
    """
    Parse a circuit string and return the root Node of the circuit tree.
    
    Parameters:
      s: str, the circuit string (e.g., "R0-p(R1,C1)-p(R2,C2)-Wo1")
      
    Returns:
      Node representing the entire circuit.
    """
    s = s.strip()
    pos = 0

    def parse_series():
        nonlocal pos
        nodes = []
        while pos < len(s):
            if s[pos] == ')':  # end of a parallel group
                break
            node = parse_term()
            nodes.append(node)
            if pos < len(s) and s[pos] == '-':
                pos += 1  # skip series separator
            else:
                break
        return nodes[0] if len(nodes) == 1 else Node('series', children=nodes)

    def parse_term():
        nonlocal pos
        if s[pos] == 'p':
            return parse_parallel()
        else:
            return parse_component()

    def parse_parallel():
        nonlocal pos
        if s[pos] != 'p':
            raise ValueError("Expected 'p' at position {}".format(pos))
        pos += 1  # skip 'p'
        if s[pos] != '(':
            raise ValueError("Expected '(' after 'p' at position {}".format(pos))
        pos += 1  # skip '('
        nodes = []
        while pos < len(s):
            node = parse_series()
            nodes.append(node)
            if pos < len(s) and s[pos] == ',':
                pos += 1  # skip comma
            elif pos < len(s) and s[pos] == ')':
                pos += 1  # skip ')'
                break
            else:
                break
        return nodes[0] if len(nodes) == 1 else Node('parallel', children=nodes)

    def parse_component():
        nonlocal pos
        match = re.match(r'[A-Za-z0-9_]+', s[pos:])
        if not match:
            raise ValueError("Expected component at position {}".format(pos))
        comp = match.group(0)
        pos += len(comp)
        return Node('component', value=comp)

    return parse_series()

def get_leaf_components(node):
    """
    Recursively retrieve a set of all leaf component names in the circuit tree.
    
    Parameters:
      node: Node, the root of the circuit tree.
      
    Returns:
      Set of strings, each a component name.
    """
    if node.node_type == 'component':
        return {node.value}
    else:
        leaves = set()
        for child in node.children:
            leaves |= get_leaf_components(child)
        return leaves

# =============================================================
# Impedance Calculation Functions
# =============================================================
def compute_impedance(node, freq, params):
    """
    Recursively compute the overall complex impedance of the circuit at a given frequency.
    
    Parameters:
      node   : Node, the root of the circuit tree.
      freq   : float, frequency in Hz.
      params : dict, mapping component names to their parameter values.
      
    Returns:
      Complex impedance of the circuit.
    """
    if node.node_type == 'component':
        return Z_component(node.value, freq, params)
    elif node.node_type == 'series':
        return sum(compute_impedance(child, freq, params) for child in node.children)
    elif node.node_type == 'parallel':
        inv_total = 0
        for child in node.children:
            Z_child = compute_impedance(child, freq, params)
            inv_total += 1 / Z_child if Z_child != 0 else 0
        return 1 / inv_total if inv_total != 0 else np.inf

# =============================================================
# Impedance Contribution Calculation
# =============================================================
def compute_impedance_contributions(tree, freq, params, factor=1.0):
    """
    Compute the absolute impedance contribution of each leaf component.
    
    For each leaf, the function perturbs its parameter by a given factor (default doubles
    the parameter value), recomputes the overall impedance, and measures the absolute change
    in impedance magnitude. This change is taken as the component's contribution.
    The contributions are normalized so that their sum equals 100%.
    
    Parameters:
      tree   : Node, the circuit tree.
      freq   : float, frequency in Hz.
      params : dict, original component parameter values.
      factor : float, relative perturbation factor (default 1.0 for a 100% increase).
      
    Returns:
      Dictionary mapping component names to their contribution percentages.
    """
    Z_base = abs(compute_impedance(tree, freq, params))
    leaves = get_leaf_components(tree)
    contributions = {}
    
    for comp in leaves:
        # Perturb the parameter for this component.
        new_params = params.copy()
        new_params[comp] = params[comp] * (1 + factor)
        Z_new = abs(compute_impedance(tree, freq, new_params))
        contributions[comp] = abs(Z_new - Z_base)
    
    total = sum(contributions.values())
    if total == 0:
        percentages = {comp: 0 for comp in contributions}
    else:
        percentages = {comp: (contrib / total) * 100 for comp, contrib in contributions.items()}
    return percentages

# =============================================================
# Main Execution: Parse Circuit, Compute Contributions, and Plot Results
# =============================================================
if __name__ == '__main__':
    # Define the circuit string.
    # Example: 'R0-p(R1,C1)-p(R2,C2)-Wo1'
    circuit_str = 'R0-p(R1,C1)-p(R2,C2)-Wo1'
    
    # Define component parameter values.
    # Adjust these values as needed for your system.
    params = {
        'R0': 100,
        'R1': 200,
        'R2': 300,
        'C1': 1e-6,
        'C2': 2e-6,
        'Wo1': 5000,
        # Example parameters for additional elements (if used)
        'CPE0': 1e-5,
        'CPE_alpha0': 0.8
    }
    
    # Choose a test frequency for computing contributions.
    test_freq = 0.001  # Hz
    
    # Parse the circuit and compute overall impedance.
    tree = parse_circuit(circuit_str)
    Z_total = compute_impedance(tree, test_freq, params)
    print("Circuit:", circuit_str)
    print("Total Impedance at {} Hz: {}".format(test_freq, Z_total))
    
    # Compute and display the impedance contributions (absolute percentages) at test_freq.
    contrib_percentages = compute_impedance_contributions(tree, test_freq, params, factor=1.0)
    print("\nImpedance Contributions (Absolute Percentages) at {} Hz:".format(test_freq))
    for comp, perc in contrib_percentages.items():
        print(f"  {comp}: {perc:.1f}%")
    
    # --- Nyquist Plot of the Impedance Spectrum ---
    # Frequency range for the Nyquist plot (from 0.001 Hz to 100 kHz)
    freqs = np.logspace(-3, 5, 150)
    Z_vals = [compute_impedance(tree, f, params) for f in freqs]
    re_Z = np.array([np.real(Z) for Z in Z_vals])
    im_Z = np.array([np.imag(Z) for Z in Z_vals])
    
    plt.figure(figsize=(8, 6))
    plt.plot(re_Z, -im_Z, 'o-', markersize=3)
    plt.xlabel("Real(Z) [Ohm]")
    plt.ylabel("-Imag(Z) [Ohm]")
    plt.title("Nyquist Plot for Circuit: " + circuit_str)
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    # --- Plot Impedance Contribution Percentages vs Frequency ---
    # For each frequency, compute the impedance contribution percentages.
    imp_percentages_vs_freq = {}
    for f in freqs:
        perc_dict = compute_impedance_contributions(tree, f, params, factor=1.0)
        for comp, perc in perc_dict.items():
            if comp not in imp_percentages_vs_freq:
                imp_percentages_vs_freq[comp] = []
            imp_percentages_vs_freq[comp].append(perc)
    
    plt.figure(figsize=(8, 6))
    for comp, perc_list in imp_percentages_vs_freq.items():
        plt.semilogx(freqs, perc_list, label=comp, marker='o', markersize=3, linestyle='-')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Impedance Contribution (%)")
    plt.title("Absolute Impedance Contribution Percentages vs Frequency")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

