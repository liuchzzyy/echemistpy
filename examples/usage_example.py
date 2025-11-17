"""
Example usage of the echemistpy package.

This script demonstrates how to use various features of the echemistpy package.
"""

import numpy as np

import echemistpy as ecp


def example_electrochemistry():
    """Example of using the Electrochemistry class."""
    print("\n=== Electrochemistry Example ===")

    # Create instance
    echem = ecp.techniques.Electrochemistry()
    print(f"Created: {echem}")

    # In a real scenario, you would load actual data:
    # data = echem.load_data('path/to/your/data.csv')
    # preprocessed = echem.preprocess()
    # results = echem.analyze()


def example_xrd():
    """Example of using the XRD class."""
    print("\n=== XRD Example ===")

    # Create instance
    xrd = ecp.techniques.XRD()
    print(f"Created: {xrd}")

    # In a real scenario:
    # data = xrd.load_data('xrd_pattern.csv')
    # results = xrd.analyze()


def example_data_processing():
    """Example of using data processing utilities."""
    print("\n=== Data Processing Example ===")

    # Create some example data
    data = np.random.randn(100) + 10

    # Normalize data
    normalized = ecp.utils.normalize(data, method="minmax")
    print(f"Original range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    # Smooth data
    noisy_data = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
    smoothed = ecp.utils.smooth(noisy_data, window_length=11, method="savgol")
    print(f"Smoothed data length: {len(smoothed)}")


def example_visualization():
    """Example of using visualization utilities."""
    print("\n=== Visualization Example ===")

    # Create example data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create line plot
    fig, ax = ecp.visualization.plot_line(
        x, y, xlabel="Time (s)", ylabel="Current (A)", title="Example Plot"
    )
    print("Created line plot")
    # fig.savefig('example_plot.png')  # Uncomment to save

    # Create heatmap
    data_2d = np.random.randn(20, 30)
    fig2, ax2 = ecp.visualization.plot_heatmap(
        data_2d,
        xlabel="X Position",
        ylabel="Y Position",
        title="Example Heatmap",
        cmap="coolwarm",
    )
    print("Created heatmap")
    # fig2.savefig('example_heatmap.png')  # Uncomment to save


def example_base_data():
    """Example of using BaseData class."""
    print("\n=== BaseData Example ===")

    # Create BaseData with numpy array
    data = np.array([[1, 2, 3], [4, 5, 6]])
    base_data = ecp.core.BaseData(data, metadata={"source": "example"})
    print(f"Created: {base_data}")

    # Convert to DataFrame
    df = base_data.to_dataframe()
    print(f"DataFrame shape: {df.shape}")

    # Convert to xarray
    xr_data = base_data.to_xarray()
    print(f"xarray shape: {xr_data.shape}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("echemistpy Package Examples")
    print(f"Version: {ecp.__version__}")
    print(f"Author: {ecp.__author__}")
    print("=" * 60)

    example_electrochemistry()
    example_xrd()
    example_data_processing()
    example_visualization()
    example_base_data()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
