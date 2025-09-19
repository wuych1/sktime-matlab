# ClaSP MATLAB Implementation

MATLAB implementation of the ClaSP (Classification Score Profile) time series segmentation algorithm, compatible with the Python reference implementation from sktime/aeon-toolkit.

## Overview

ClaSP is a time series segmentation algorithm that identifies change points by analyzing classification scores across sliding windows. This MATLAB implementation provides equivalent functionality to the Python version with matching results.

## Features

- **ClaSPTransformer**: Computes the classification score profile for time series
- **ClaSPSegmenter**: Identifies change points using recursive segmentation
- Full compatibility with Python sktime implementation
- Comprehensive test suite for validation

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

Basic usage example:

```matlab
% Create ClaSP segmenter
clasp = ClaSPSegmenter('PeriodLength', 30, 'NumChangePoints', 2);

% Find change points in time series
changePoints = clasp.fitPredict(timeSeries);

% Access scores and profiles
scores = clasp.Scores;
profiles = clasp.Profiles;
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
1. Computes k-nearest neighbors for each sliding window
2. Creates binary classification problems at each potential split point
3. Evaluates split quality using ROC-AUC scores
4. Recursively segments the time series using a priority queue

## References

- [sktime ClaSP implementation](https://github.com/aeon-toolkit/aeon/blob/main/aeon/segmentation/_clasp.py)
- [Original ClaSP paper](https://link.springer.com/article/10.1007/s10618-023-00988-8)

## License

This implementation follows the same BSD 3-Clause License as sktime.