# ClaSP MATLAB Implementation

MATLAB implementation of the ClaSP (Classification Score Profile) time series segmentation algorithm, functionally equivalent to the Python reference implementation from sktime/aeon-toolkit.

## Overview

ClaSP is a time series segmentation algorithm that identifies change points by analyzing classification scores across sliding windows. The algorithm uses k-nearest neighbors classification and ROC-AUC scoring to detect optimal split points in time series data.

## Features

- **ClaSPTransformer**: Computes the classification score profile using k-NN and ROC-AUC
- **ClaSPSegmenter**: Identifies change points using recursive priority-queue segmentation
- **Boundary-aware**: Properly handles exclusion zones at series start/end to avoid spurious detections
- **Parameter flexibility**: Adjustable window size, exclusion radius, and number of change points
- **Validated implementation**: Tested against Python sktime with real-world datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wuych1/sktime-matlab.git
cd sktime-matlab
```

2. Add the segmentation folder to your MATLAB path:
```matlab
addpath(genpath('segmentation'));
```

## Usage

### Basic Example

```matlab
% Create ClaSP segmenter
clasp = ClaSPSegmenter('PeriodLength', 30, 'NumChangePoints', 2);

% Find change points in time series
changePoints = clasp.fitPredict(timeSeries);

% Access scores and profiles
scores = clasp.Scores;
profiles = clasp.Profiles;
```

### Important Parameters

- **PeriodLength**: Window size for sliding window (typically 10-50)
- **NumChangePoints**: Number of change points to detect
- **ExclusionRadius**: Fraction of series length to exclude around change points (default 0.05)
  - Use 0.05-0.1 for general cases
  - Reduce to 0.02 or less if boundary changes are expected

### Handling Boundary Effects

The algorithm excludes positions near the start and end of the series to avoid spurious boundary detections. If you need to detect changes near boundaries:

```matlab
% Use smaller exclusion radius for boundary detection
clasp = ClaSPSegmenter('PeriodLength', 20, ...
                        'NumChangePoints', 3, ...
                        'ExclusionRadius', 0.02);
```

## Testing

Run the comprehensive comparison tests:

```matlab
cd testing
run('comprehensive_comparison.m')
```

Compare with Python implementation:
```bash
cd testing/python
python comprehensive_test.py
```

## Algorithm Details

The ClaSP algorithm:
1. Computes k-nearest neighbors for each sliding window using z-normalized Euclidean distance
2. Creates binary classification problems at each potential split point
3. Evaluates split quality using ROC-AUC scores
4. Recursively segments the time series using a priority queue
5. Applies exclusion zones to prevent trivial matches and boundary artifacts

### Key Implementation Features

- **Boundary Exclusion**: Automatically excludes change points within `exclusionRadius * length(series)` of the start/end
- **Profile Interpolation**: Uses linear interpolation for interior gaps, low values for boundaries
- **Trivial Match Prevention**: Ensures change points are sufficiently separated
- **Normalized Distance**: Uses z-normalization for robust distance computation

## Performance Notes

- For large datasets (>10,000 points), computation can take several minutes
- Consider using smaller window sizes (10-20) for faster processing
- The algorithm's complexity is O(n²) for k-NN computation

## References

- [sktime ClaSP implementation](https://github.com/aeon-toolkit/aeon/blob/main/aeon/segmentation/_clasp.py)
- [Original ClaSP paper](https://link.springer.com/article/10.1007/s10618-023-00988-8)
- Schäfer, P., Ermshaus, A., & Leser, U. (2021). ClaSP - Time Series Segmentation. CIKM.

## License

This implementation follows the same BSD 3-Clause License as sktime.