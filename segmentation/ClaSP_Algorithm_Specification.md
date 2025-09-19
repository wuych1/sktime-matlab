# ClaSP Algorithm Specification

Based on detailed analysis of the Python reference implementation from aeon-toolkit.

## Overview

ClaSP (Classification Score Profile) is a time series segmentation algorithm that detects change points by:
1. Creating sliding windows of the time series
2. Finding k-nearest neighbors for each window
3. Evaluating binary classification performance at each potential split point
4. Using a priority queue to recursively find multiple change points

## Core Algorithm Components

### 1. ClaSP Transformer (`clasp()` function)

**Input:**
- `X`: Time series of length `n`
- `m`: Window size
- `k`: Number of nearest neighbors (default: 3)

**Output:**
- `profile`: Classification score profile of length `n-m+1`
- `knn_mask`: k-NN indices matrix of size `(k, n-m+1)`

**Algorithm Steps:**

#### Step 1: Extract Sliding Windows
```python
# Using numpy stride tricks to create sliding windows
windows = _sliding_window(X, m)  # Shape: (n-m+1, m)
```

#### Step 2: Compute k-Nearest Neighbors
```python
knn_mask = _compute_distances_iterative(X, m, k)
# Returns: (k, n-m+1) matrix where knn_mask[i,j] contains the i-th nearest neighbor index for window j
```

**Distance Computation Details:**
1. For each window `i`, compute sliding dot product with all other windows
2. Calculate normalized distance using formula:
   ```
   dist = 2 * m * (1 - (dot_product - m * mean_i * mean_j) / (m * std_i * std_j))
   ```
3. Apply exclusion zone to prevent trivial matches (within `m/4` indices)
4. Select k smallest distances

#### Step 3: Compute Classification Profile
```python
profile = _calc_profile(m, knn_mask, score_function, exclusion_zone)
```

**Profile Computation Details:**
For each potential split point `split_idx` from `exclusion_zone` to `n_timepoints - exclusion_zone`:

1. **Create Binary Labels:**
   ```python
   y_true = [0, 0, ..., 0, 1, 1, ..., 1]  # 0 for indices < split_idx, 1 for >= split_idx
   ```

2. **Generate k-NN Predictions:**
   ```python
   # For each window j, look at its k nearest neighbors
   for j in range(n_timepoints):
       neighbor_labels = y_true[knn_mask[:, j]]  # Labels of k neighbors
       y_pred[j] = 1 if sum(neighbor_labels) > k/2 else 0  # Majority vote
   ```

3. **Apply Exclusion Zone:**
   ```python
   # Force predictions around split point to be 1 (right side)
   y_pred[split_idx-m:split_idx] = 1
   ```

4. **Compute ROC-AUC Score:**
   ```python
   profile[split_idx] = roc_auc_score(y_true, y_pred)
   ```

#### Step 4: Interpolate Profile
```python
# Fill NaN values using pandas interpolation
profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
```

### 2. ClaSP Segmentation (`_segmentation()` function)

**Input:**
- `X`: Time series
- `clasp`: ClaSP transformer instance
- `n_change_points`: Number of change points to find
- `exclusion_radius`: Minimum distance between change points (as fraction)

**Output:**
- `change_points`: List of detected change point indices
- `scores`: List of scores for each change point
- `profiles`: List of profiles used

**Algorithm Steps:**

#### Step 1: Initialize Priority Queue
```python
# Compute global profile
profile = clasp.transform(X)

# Add to priority queue with negative max score (for max-heap behavior)
queue.put((-np.max(profile), [range(0, len(X)), np.argmax(profile), profile]))
```

#### Step 2: Iterative Change Point Detection
```python
for i in range(n_change_points):
    if queue.empty():
        break

    # Get highest scoring segment
    priority, (profile_range, change_point, full_profile) = queue.get()

    # Store results
    change_points.append(change_point)
    scores.append(-priority)
    profiles.append(full_profile)

    # Split into left and right ranges
    left_range = range(profile_range[0], change_point)
    right_range = range(change_point, profile_range[-1])

    # Process both sub-ranges
    for ranges in [left_range, right_range]:
        exclusion_zone = int(len(ranges) * exclusion_radius)

        # Skip if range too small
        if len(ranges) - clasp.window_length <= 2:
            continue

        # Extract sub-series and compute local profile
        sub_series = X[ranges]
        local_profile = clasp.transform(sub_series)

        # Find best change point in local profile
        valid_profile = local_profile.copy()
        valid_profile[:exclusion_zone] = -np.inf
        valid_profile[-exclusion_zone:] = -np.inf

        if np.max(valid_profile) > -np.inf:
            local_max_idx = np.argmax(valid_profile)
            global_change_point = ranges[0] + local_max_idx

            # Check if not trivial match
            if not _is_trivial_match(global_change_point, change_points, exclusion_radius, len(X)):
                queue.put((-np.max(valid_profile), [ranges, local_max_idx, local_profile]))
```

#### Step 3: Trivial Match Detection
```python
def _is_trivial_match(new_cp, existing_cps, exclusion_radius, series_length):
    exclusion_zone = int(series_length * exclusion_radius)
    for existing_cp in existing_cps:
        if abs(new_cp - existing_cp) < exclusion_zone:
            return True
    return False
```

## Key Mathematical Formulations

### 1. Normalized Distance Between Windows
```
dist_ij = 2 * m * (1 - (dot_product_ij - m * mean_i * mean_j) / (m * std_i * std_j))
```

### 2. Binary Classification Setup
- **True labels**: `y_true[i] = 0` if `i < split_idx`, else `1`
- **Predicted labels**: Majority vote of k-nearest neighbors' true labels
- **Score**: ROC-AUC of the binary classification problem

### 3. ROC-AUC Computation
```python
def roc_auc_score(y_true, y_pred):
    # Sort by prediction scores
    sorted_indices = np.argsort(-y_pred)
    sorted_labels = y_true[sorted_indices]

    # Compute TPR and FPR
    tpr = np.cumsum(sorted_labels) / np.sum(sorted_labels)
    fpr = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)

    # Add (0,0) point and compute AUC
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return np.trapz(tpr, fpr)
```

## Critical Implementation Details

1. **Exclusion Zones**: Multiple exclusion zones are used:
   - k-NN computation: `m/4` to avoid trivial matches
   - Profile computation: `max(m, series_length * exclusion_radius)`
   - Segmentation: `series_length * exclusion_radius`

2. **Priority Queue**: Uses negative scores to implement max-heap behavior

3. **Range Management**: Careful index translation between local and global coordinates

4. **Edge Cases**: Proper handling of short series, insufficient neighbors, and boundary effects

5. **Performance**: Uses FFT for sliding dot products and Numba for acceleration

## MATLAB Implementation Requirements

1. Implement sliding window extraction using efficient indexing
2. Use FFT-based convolution for dot product computation
3. Implement priority queue using cell arrays with score-based sorting
4. Handle all exclusion zone logic correctly
5. Ensure proper index translation between local and global coordinates
6. Implement robust ROC-AUC calculation
7. Add comprehensive error handling for edge cases