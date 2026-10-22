# NumPy Scientific Computing & Array Rules Reference

This document details mandatory NumPy conventions and error-prevention guidelines for scientific Python workflows.

---

## 1. Truth Values of NumPy Arrays (`ValueError: Ambiguous Truth Value`)

### The Error
Evaluating a NumPy array directly in a boolean context or inside Python's built-in `any()` / `all()` raises:
```
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

### Problematic Anti-Patterns
```python
# [X] BUG: Generator expression yields 1D boolean array for each row
if any(distance < 1e-5 for distance in distances):
    ...

# [X] BUG: Direct conditional on array
if distances < 1e-5:
    ...

# [X] BUG: Direct truth check on array
if array:
    ...
```

### Correct Vectorized Formulations
```python
# [V] Check if ANY element satisfies condition:
if np.any(distances < 1e-5):
    ...
# or equivalently:
if (distances < 1e-5).any():
    ...

# [V] Check if ALL elements satisfy condition:
if np.all(distances >= 0):
    ...
# or equivalently:
if (distances >= 0).all():
    ...
```

---

## 2. Vectorized Boundary and Mask Filtering

### Problem: Duplicate Appends via Sequential `if` Blocks
```python
# [X] ANTI-PATTERN: Produces duplicate entries for atoms near multiple faces (edges/corners)
boundary_indices = []
for i, pos in enumerate(scaled_positions):
    if pos[0] < tol or pos[0] > (1 - tol): boundary_indices.append(i)
    if pos[1] < tol or pos[1] > (1 - tol): boundary_indices.append(i) # DUPLICATE!
    if pos[2] < tol or pos[2] > (1 - tol): boundary_indices.append(i) # DUPLICATE!
```

### Solution: Single Vectorized Boolean Mask
```python
# [V] CANONICAL VECTORIZED PATTERN (Guarantees zero duplicate indices)
scaled_positions = atoms.get_scaled_positions(wrap=True)
tol = 0.05

# Condition across all 3 axes:
is_near_boundary = np.any((scaled_positions < tol) | (scaled_positions > (1.0 - tol)), axis=1)

# Extract unique indices:
boundary_indices = np.where(is_near_boundary)[0]
```

---

## 3. Masking and Indexing Best Practices

- **Find indices satisfying conditions**:
  ```python
  ti_indices = np.where(symbols == "Ti")[0]
  surface_ti = np.where((symbols == "Ti") & (positions[:, 2] > z_cutoff))[0]
  ```
- **Exclude Self-Interactions in Distance Matrices**:
  ```python
  dists = atoms.get_all_distances(mic=True)
  np.fill_diagonal(dists, np.inf)
  min_interatomic_distance = np.min(dists)
  ```
