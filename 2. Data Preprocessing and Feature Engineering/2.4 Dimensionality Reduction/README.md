# Dimensionality Reduction Techniques

Dimensionality reduction is an essential technique in data preprocessing and visualization. It helps in reducing the number of random variables under consideration by obtaining a set of principal variables. This document provides an overview of some commonly used dimensionality reduction techniques, including PCA, t-SNE, and LDA, along with examples and visualizations.

## Table of Contents

- **Principal Component Analysis (PCA)**
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
- **Linear Discriminant Analysis (LDA)**

## Introduction

Dimensionality reduction techniques are used to reduce the number of features in a dataset while retaining as much information as possible. These techniques are useful for:
- Simplifying models to reduce computational costs.
- Mitigating the curse of dimensionality.
- Visualizing high-dimensional data.

## Read here :
- **<a href="https://www.analyticsvidhya.com/blog/2021/09/pca-and-its-underlying-mathematical-principles/">PCA And It’s Underlying Maths</a>**
- **<a href="https://towardsdatascience.com/t-sne-behind-the-math-4d213b9ebab8#:~:text=t%2DSNE%20tries%20to%20figure,in%20lower%20dimensions%20as%20well.">Maths behind t-SNE </a>**
- **<a href="https://omarshehata.github.io/lda-explorable/">A Geometric Intuition for LDA</a>**
## Watch here :
- **<a href="https://www.youtube.com/watch?v=FgakZw6K1QQ">PCA -Statsquest</a>**
- **<a href="https://www.youtube.com/watch?v=g-Hb26agBFg">PCA - Serrano Academy</a>**
- **<a href="https://www.youtube.com/watch?v=fkf4IBRSeEc">PCA - Steve Burnton</a>**
- **<a href="https://www.youtube.com/playlist?list=PLWhu9osGd2dB9uMG5gKBARmk73oHUUQZS">SVD series - Visual Kernel</a>**

## Principal Component Analysis (PCA)

PCA is a linear dimensionality reduction technique that transforms data into a new coordinate system. The new coordinates (principal components) are ordered such that the first few retain most of the variation present in the original dataset.

### Steps to Perform PCA

1. Standardize the data.
2. Compute the covariance matrix.
3. Compute the eigenvectors and eigenvalues of the covariance matrix.
4. Project the data onto the new feature space.

### Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA results
plt.figure(figsize=(10, 6))
for target, color, label in zip(np.unique(y), ['r', 'g', 'b'], target_names):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], color=color, label=label, alpha=0.7)
plt.title('Iris Data after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

```
<hr>
<hr>

## t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a nonlinear dimensionality reduction technique that is well-suited for embedding high-dimensional data into a low-dimensional space (typically 2D or 3D) for visualization. It focuses on preserving local structure.

### Steps to Perform t-SNE

1. Standardize the data.
2. Apply t-SNE to reduce to 2 or 3 dimensions.
3. Visualize the results.

### Example

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot the t-SNE results
plt.figure(figsize=(10, 6))
for target, color, label in zip(np.unique(y), ['r', 'g', 'b'], target_names):
    plt.scatter(X_tsne[y == target, 0], X_tsne[y == target, 1], color=color, label=label, alpha=0.7)
plt.title('Iris Data after t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()
```

<hr>
<hr>

## Linear Discriminant Analysis (LDA)

LDA is used for classification and dimensionality reduction in supervised learning. It finds the linear combinations of features that best separate different classes.

### Steps to Perform LDA

1. Standardize the data.
2. Compute the LDA components.
3. Transform the data.
4. Visualize the results.

### Example

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Plot the LDA results
plt.figure(figsize=(10, 6))
for target, color, label in zip(np.unique(y), ['r', 'g', 'b'], target_names):
    plt.scatter(X_lda[y == target, 0], X_lda[y == target, 1], color=color, label=label, alpha=0.7)
plt.title('Iris Data after LDA')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.grid(True)
plt.show()


```
