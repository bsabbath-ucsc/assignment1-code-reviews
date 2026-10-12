# CSE 242 Assignment 1

# Author: Brian Sabbath.

import numpy as np
import csv

from pathlib import Path

# For P3 Q2:

# Generate independent samples from N([2,-1], [[3,1.2],[1.2,1]]).
# The population correlation is 1.2/sqrt(3*1) = 0.6928, so the point
# cloud should slope upward. A generator makes the sample reproducible.

def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    # Return reproducible Gaussian samples of shape (n_samples, 2).
    location = np.asarray([2, -1] if mean is None else mean, dtype=float)
    cov = np.asarray(
        [[3, 1.2], [1.2, 1]] if covariance is None else covariance,
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(location, cov, size=n_samples)


# For P3 Q3:

# 1. Subtract each feature's sample mean to form centered observations.
# 2. Estimate Sigma with the sample covariance matrix.
# 3. Since S is symmetric, eigh should return orthonormal eigenvectors.
# 4. Sort descending: maximizing (u^T)*S*u under (u^T)*u=1 selects the
#    eigenvector with the largest eigenvalue. 
# 5. Scores = Z * components
def pca(X):
    # Fit PCA with NumPy and return all intermediate arrays.

    # X has shape (n, 2), n >= 2. Components are orthonormal columns,
    # eigenvalues descend, and covariance uses divisor n-1. 

    samples = np.asarray(X, dtype=float)
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order]
    scores = centered @ components
    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores,
    }


# For P3 Q4:

# The variance along component i equals lambda i (subscripted. Not sure
# if there's a '^' equivalent for subscripts). Total variance is
# trace(S)=sum(lambda i), so its fraction is lambda i / sum(lambda j).

def explained_variance(eigenvalues):
    # Return same-shape fractions for nonnegative values of positive sum.
    values = np.asarray(eigenvalues, dtype=float)
    return values / np.sum(values)


# Excel data for P3:

# Figure 1: raw samples on top, principal directions on bottom.
# Figure 2: principal component scores on top, explained variance on bottom.
# Figure 3: original samples and their reconstruction using only PC1.

def export_excel_data(X, result, output_directory):
    #Export the data for Figures 1, 2, and 3 as five Excel-ready CSVs.

    # Original feature values are in figure_1_raw_data.csv. The other files
    # contain calculated PCA values needed for the remaining plots.
    # Variance fractions are formatted as percentages in Excel.
    # Return the saved file paths without changing X or result.

    samples = np.asarray(X, dtype=float)
    mean = result["mean"]
    values = result["eigenvalues"]
    components = result["components"]
    scores = result["scores"]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    def write_csv(filename, header, rows):

        path = output_directory / filename
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        saved_paths.append(path)

    # Figure 1, top: 
    # Feature_1 is x and Feature_2 is y.
    write_csv(
        "figure_1_raw_data.csv",
        ["Sample", "Feature_1", "Feature_2"],
        ((index + 1, *point) for index, point in enumerate(samples)),
    )

    # Figure 1, bottom: 
    # Add these two direction series to the raw plot.
    displacement = components * (2.0 * np.sqrt(values))
    positions = (
        ("Negative endpoint", -1),
        ("Mean", 0),
        ("Positive endpoint", 1),
    )
    direction_rows = []
    for label, factor in positions:
        pc1_point = mean + factor * displacement[:, 0]
        pc2_point = mean + factor * displacement[:, 1]
        direction_rows.append((label, *pc1_point, *pc2_point))
    write_csv(
        "figure_1_principal_directions.csv",
        ["Position", "PC1_x", "PC1_y", "PC2_x", "PC2_y"],
        direction_rows,
    )

    # Figure 2, top: 
    # PC1_score is x and PC2_score is y.
    write_csv(
        "figure_2_scores.csv",
        ["Sample", "PC1_score", "PC2_score"],
        ((index + 1, *point) for index, point in enumerate(scores)),
    )

    # Figure 2, bottom: 
    # Component labels and Variance_fraction bar heights.
    fractions = (
        explained_variance(values) if np.sum(values) > 0 else [None, None]
    )
    write_csv(
        "figure_2_explained_variance.csv",
        ["Component", "Eigenvalue", "Variance_fraction"],
        ((f"PC{index + 1}", value, fractions[index])
         for index, value in enumerate(values)),
    )

    # Figure 3: plot the original and reconstructed feature pairs.
    reconstruction = mean + np.outer(scores[:, 0], components[:, 0])
    write_csv(
        "figure_3_reconstruction.csv",
        [
            "Sample", "Original_Feature_1", "Original_Feature_2",
            "Reconstructed_Feature_1", "Reconstructed_Feature_2",
        ],
        ((index + 1, *point, *reconstruction[index])
         for index, point in enumerate(samples)),
    )
    return saved_paths


if __name__ == "__main__":

    data = generate_data()
    fitted = pca(data)
    print("Sample mean:\n", fitted["mean"])
    print("Sample covariance:\n", fitted["covariance"])
    print("Sample correlation:", np.corrcoef(data, rowvar=False)[0, 1])
    print("Eigenvalues:", fitted["eigenvalues"])
    print("Components (columns):\n", fitted["components"])
    print("Explained variance:", explained_variance(fitted["eigenvalues"]))
    reconstructed = fitted["mean"] + np.outer(
        fitted["scores"][:, 0], fitted["components"][:, 0],
    )
    squared_error = np.sum((data - reconstructed) ** 2, axis=1)
    print("PC1 mean squared reconstruction distance:", np.mean(squared_error))

    output_directory = Path(__file__).resolve().parent / "asgn1_excel_data"
    saved_files = export_excel_data(data, fitted, output_directory)
    print("\nSaved Excel data:")
    for saved_file in saved_files:
        print(saved_file)
