"""
Plotting Module

Functions for visualizing CDP results.
"""

import matplotlib.pyplot as plt
from itertools import cycle


def plot_curve(x, y, title, xlabel, ylabel, var, mode='strain_rate', style=None):
    """
    Plot a single curve with appropriate labeling.

    Args:
        x: X-axis data
        y: Y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        var: Variable value (strain rate or temperature)
        mode: 'strain_rate' or 'temperature' to determine label format
        style: Optional style dictionary
    """
    if style is None:
        style = {"color": "#1f77b4", "marker": "o"}

    # Create label based on mode
    if mode == 'temperature':
        label = r'$T=$' + str(int(var)) + ' [°C]'
    else:
        label = r'$\dot{\varepsilon}=$' + str(var) + ' [s$^{-1}$]'

    plt.plot(x, y, label=label, **style, linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_multiple_curves(x, y, title, xlabel, ylabel, var, mode='strain_rate'):
    """
    Create a figure with multiple curves.

    Args:
        x: X-axis data (single array or list of arrays)
        y: List of Y-axis data arrays
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        var: List of variable values (strain rates or temperatures)
        mode: 'strain_rate' or 'temperature' to determine label format
    """
    custom_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#a55194", "#393b79"
    ]
    custom_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '<', '>', '8']
    custom_style = cycle([
        {"color": c, "marker": m}
        for c, m in zip(custom_colors, custom_markers)
    ])

    plt.figure()

    # Check if x is a list of arrays or single array
    if len(x) != len(y):
        # Single x-array for all curves
        for i in range(len(y)):
            style = next(custom_style)
            plot_curve(x, y[i], title, xlabel, ylabel, var[i], mode, style)
    else:
        # Different x-array for each curve
        for i in range(len(y)):
            style = next(custom_style)
            plot_curve(x[i], y[i], title, xlabel, ylabel, var[i], mode, style)

    plt.grid()
    plt.legend(loc="upper right")
    plt.show()


def plot_all_results(results, var, mode='strain_rate'):
    """
    Plot all CDP results including compression, tension, and damage.

    Args:
        results: Dictionary of CDP results from calculate_stress_strain or calculate_stress_strain_temp
        var: List of variable values (strain rates or temperatures)
        mode: 'strain_rate' or 'temperature' to determine label format
    """
    # Plot compression
    if mode == 'temperature':
        # Use temperature-specific strain arrays
        plot_multiple_curves(
            results['compression']['strain temp'],
            results['compression']['stress'],
            'Compressive Strain - Compressive Stress',
            'Compressive Strain [-]',
            'Compressive Stress [MPa]',
            var,
            mode
        )
    else:
        # Use single strain array for all rates
        plot_multiple_curves(
            results['compression']['strain'],
            results['compression']['stress'],
            'Compressive Strain - Compressive Stress',
            'Compressive Strain [-]',
            'Compressive Stress [MPa]',
            var,
            mode
        )

    # Plot inelastic compression
    plot_multiple_curves(
        results['compression']['inelastic strain'],
        results['compression']['inelastic stress'],
        'Compressive Inelastic Strain - Compressive Stress',
        'Compressive Inelastic Strain [-]',
        'Compressive Stress [MPa]',
        var,
        mode
    )

    # Plot compression damage
    plt.figure()
    plot_curve(
        results['compression']['inelastic strain'][0],
        results['compression']['damage'],
        'Compressive Damage',
        'Compressive Inelastic Strain [-]',
        'Damage [-]',
        var[0],
        mode
    )
    plt.grid()
    plt.show()

    # Plot tension
    plot_multiple_curves(
        results['tension']['crack opening'],
        results['tension']['stress'],
        'Crack Opening - Tensile Stress (Bilinear)',
        'Crack Opening [mm]',
        'Cracking Stress [MPa]',
        var,
        mode
    )
    plot_multiple_curves(
        results['tension']['crack opening'],
        results['tension']['stress exponential'],
        'Crack Opening - Tensile Stress (Power Law)',
        'Crack Opening [mm]',
        'Cracking Stress [MPa]',
        var,
        mode
    )
    plot_multiple_curves(
        results['tension']['cracking strain'],
        results['tension']['stress'],
        'Cracking Strain - Tensile Stress',
        'Cracking Strain [-]',
        'Cracking Stress [MPa]',
        var,
        mode
    )
    plot_multiple_curves(
        results['tension']['cracking strain'],
        results['tension']['stress exponential'],
        'Cracking Strain - Tensile Stress (Power Law)',
        'Cracking Strain [-]',
        'Cracking Stress [MPa]',
        var,
        mode
    )

    # Plot tension damage
    plt.figure()
    plot_curve(
        results['tension']['cracking strain'][0],
        results['tension']['damage'],
        'Tension Damage (Bilinear)',
        'Cracking Strain [-]',
        'Damage [-]',
        var[0],
        mode
    )
    plt.grid()
    plt.show()

    plt.figure()
    plot_curve(
        results['tension']['cracking strain'][0],
        results['tension']['damage exponential'],
        'Tension Damage (Power Law)',
        'Cracking Strain [-]',
        'Damage [-]',
        var[0],
        mode
    )
    plt.grid()
    plt.show()
