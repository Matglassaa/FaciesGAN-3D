"""
This script contains custom plotting configurations and colormaps used across the visualization scripts.
It ensures consistent styling and facies representations for all generated plots.

Example usage on a Linux cluster:
    nohup python -u custom_plots.py > custom_plots.log 2>&1 &
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Custom plotting configurations and colormaps test script.")
    return parser.parse_args()

class FaciesColorMap(mcolors.ListedColormap):
    """
    Custom colormap for facies visualization based on the provided color scheme.
    Each facies type is assigned a specific color for clear differentiation in plots.
    """
    FACIES_PROPERTIES = {
        'undefined':           {'val': 0,  'color': "#ff0000", 'description': "Background/Undefined"},
        'channel_lag':         {'val': 1,  'color': "#f1970f", 'description': "Active channel fill, coarse-grained"},
        'point_bar':           {'val': 2,  'color': "#f3dd12", 'description': "Lower energy channel margins"},
        'sand_plug':           {'val': 3,  'color': "#af8f00", 'description': "Fine-grained oxbow/plug fill"},
        'crevasse_splay_core': {'val': 4,  'color': "#fffc65", 'description': "Proximal high-energy splay"},
        'crevasse_channel':    {'val': 5,  'color': "#ffd986", 'description': "Feeder channel for splays"},
        'crevasse_splay_delta':{'val': 6,  'color': "#ff9853", 'description': "Distal fan-like splay deposit"},
        'levee':               {'val': 7,  'color': "#27ae60", 'description': "Sand/silt ridges bordering channel"},
        'overbank':            {'val': 8,  'color': "#33ff00", 'description': "Stabilized/vegetated levee"},
        'mud_plug':            {'val': 9,  'color': "#fff7db", 'description': "Fine silts/clays far from channel"},
        'hemipelagic_plug':    {'val': 10, 'color': "#7a7d80", 'description': "Silts near active channel belts"},
        'wetland':             {'val': 11, 'color': "#d862f0", 'description': "Organic rich, very fine sediment"},
        'draping':             {'val': 12, 'color': "#8dd5e7", 'description': "Lateral accretion sand bodies"},
        'pelagic':             {'val': 13, 'color': "#3498db", 'description': "Lacustrine clay/silt"}
    }

    def __init__(self):
        sorted_facies = sorted(self.FACIES_PROPERTIES.values(), key=lambda x: x['val'])
        colors = [props['color'] for props in sorted_facies]
        super().__init__(colors, name='FaciesColorMap')

    @classmethod
    def get_norm(cls):
        """
        Creates a BoundaryNorm mapped strictly to the discrete facies integers.
        This prevents Matplotlib from interpolating colors.
        """
        vals = [v['val'] for v in cls.FACIES_PROPERTIES.values()]
        boundaries = np.arange(min(vals) - 0.5, max(vals) + 1.5, 1)
        return mcolors.BoundaryNorm(boundaries, len(vals))

def apply_custom_plotting_flavor():
    """
    Applies the custom plotting style inspired by the supervisor's notebook.
    Features thin lines, grey ticks, and no top/right spines.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['font.size'] = 8

    plt.rcParams['figure.constrained_layout.use'] = False

    # Figure Title (Suptitle) formatting
    plt.rcParams['figure.titlesize'] = 10       # Slightly larger than axes title
    plt.rcParams['figure.titleweight'] = 'bold'  # Makes the main title stand out

    # Axes Title formatting
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.titleweight'] = 'normal'  # 'bold', 'light', or 'normal'
    plt.rcParams['axes.titlelocation'] = 'center'  # 'left', 'center', or 'right'
    plt.rcParams['axes.titlepad'] = 6.0          # Padding between title and top spine

    # Axis and edge coloring
    plt.rcParams['axes.edgecolor'] = '#8C8C8C'
    plt.rcParams['axes.labelcolor'] = '#595959'
    plt.rcParams['axes.linewidth'] = 0.4
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    
    # Tick formatting
    plt.rcParams['xtick.color'] = '#8C8C8C'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['xtick.major.pad'] = 3
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 0.4
    plt.rcParams['xtick.minor.pad'] = 3
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['xtick.minor.width'] = 0.4
    
    plt.rcParams['ytick.color'] = '#8C8C8C'
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['ytick.major.pad'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 0.4
    plt.rcParams['ytick.minor.pad'] = 3
    plt.rcParams['ytick.minor.size'] = 1.5
    plt.rcParams['ytick.minor.width'] = 0.4

    # Grid formatting
    plt.rcParams['grid.color'] = '#BFBFBF'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linewidth'] = 0.2

    # Line formatting
    plt.rcParams['lines.linewidth'] = 0.6

def main():
    args = parse_args()

if __name__ == "__main__":
    main()