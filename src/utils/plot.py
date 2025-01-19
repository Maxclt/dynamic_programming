import numpy as np
import matplotlib.pyplot as plt


def compare_approx(
    x_values, curve1, curve2, curve1_label="Curve 1", curve2_label="Curve 2"
):
    """
    Plots two curves and shows the maximum distance between them.

    Parameters:
        x_values (array-like): The x-axis values corresponding to the curves.
        curve1 (array-like): The first curve (y-values).
        curve2 (array-like): The second curve (y-values).
        curve1_label (str): Label for the first curve.
        curve2_label (str): Label for the second curve.
    """
    # Compute the absolute difference and find the maximum distance
    abs_difference = np.abs(np.array(curve1) - np.array(curve2))
    max_distance = np.max(abs_difference)
    max_index = np.argmax(abs_difference)  # Find the index where max difference occurs
    max_x = x_values[max_index]  # Get the corresponding x-coordinate

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot the curves
    ax.plot(
        x_values,
        curve1,
        lw=2,
        alpha=0.6,
        label=curve1_label,
    )
    ax.plot(
        x_values,
        curve2,
        lw=2,
        alpha=0.6,
        label=curve2_label,
    )

    # Plot the max distance as a vertical line
    ax.axvline(
        x=max_x,
        color="red",
        linestyle="--",
        alpha=0.8,
    )
    ax.annotate(
        f"Max distance\n{max_distance:.5f}",
        xy=(max_x, max(curve1[max_index], curve2[max_index])),
        xytext=(max_x + 0.5, max(curve1) - 1),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        fontsize=10,
        ha="left",
    )

    # Add labels and legend
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend(loc="lower right")

    # Show the plot
    plt.show()
