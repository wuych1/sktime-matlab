# ClaSP Time Series Segmentation for MATLAB

This package provides a MATLAB implementation of the ClaSP (Classification Score Profile) time series segmentation algorithm, designed to detect change points in time series data without requiring prior knowledge of the number or location of change points.

## Overview

ClaSP is a domain-agnostic time series segmentation method that:
- Uses k-nearest neighbors and binary classification to identify change points
- Computes a profile of classification scores across the time series
- Supports recursive segmentation to find multiple change points
- Includes automatic window size selection using FFT analysis

## Quick Start

```matlab
% Add the segmentation folder to your MATLAB path
addpath(genpath('segmentation/'));

% Generate sample time series
ts = [sin(0.1*(1:300)), 2*sin(0.3*(301:600)), 0.5*cos(0.05*(601:1000))]';

% Create ClaSP segmenter (Python-like API)
clasp = ClaSPSegmenter('PeriodLength', 50, 'NumChangePoints', 2);

% Fit and predict change points
foundCps = clasp.fitPredict(ts);

% Access results
profiles = clasp.Profiles;
scores = clasp.Scores;

fprintf('The found change points are: %s\n', mat2str(foundCps'));
```

## API Reference

### ClaSPSegmenter Class

Main class for time series segmentation using the ClaSP algorithm.

#### Constructor
```matlab
clasp = ClaSPSegmenter('Name', Value, ...)
```

**Parameters:**
- `PeriodLength` (default: 10) - Window size for sliding analysis
- `NumChangePoints` (default: 1) - Number of change points to detect
- `ExclusionRadius` (default: 0.05) - Minimum distance between change points as fraction of series length

#### Methods

##### fitPredict(timeSeries)
Fit the ClaSP model and predict change points.

**Input:**
- `timeSeries` - Input time series data (numeric vector)

**Output:**
- `changePoints` - Detected change point indices (vector)

##### findDominantWindowSizes(timeSeries)
Automatically determine optimal window sizes using FFT analysis.

**Input:**
- `timeSeries` - Input time series data (numeric vector)

**Output:**
- `windowSizes` - Optimal window size(s)

#### Properties

##### Public Properties
- `PeriodLength` - Window size for sliding analysis
- `NumChangePoints` - Number of change points to detect
- `ExclusionRadius` - Exclusion radius for avoiding trivial matches

##### Read-only Properties
- `Profiles` - Computed classification score profiles
- `Scores` - Change point scores

## File Structure

```
segmentation/
├── ClaSPSegmenter.m          # Main segmentation class
├── ClaSPTransformer.m        # Core transformation logic
├── utils/
│   ├── slidingDotProduct.m   # Efficient sliding window operations
│   ├── computeKnnDistances.m # K-nearest neighbor computation
│   └── calculateRocAuc.m     # ROC-AUC calculation for scoring
├── examples/
│   └── claspExample.m        # Usage examples and demonstrations
└── README.md                 # This file
```

## Examples

### Basic Usage
```matlab
% Simple segmentation
clasp = ClaSPSegmenter('PeriodLength', 20, 'NumChangePoints', 3);
changePoints = clasp.fitPredict(timeSeries);
```

### Automatic Window Size Selection
```matlab
% Let ClaSP find optimal window size
clasp = ClaSPSegmenter('NumChangePoints', 5);
optimalWindow = clasp.findDominantWindowSizes(timeSeries);
clasp.PeriodLength = optimalWindow;
changePoints = clasp.fitPredict(timeSeries);
```

### Advanced Usage with Profile Analysis
```matlab
clasp = ClaSPSegmenter('PeriodLength', 30, 'NumChangePoints', 4);
foundCps = clasp.fitPredict(timeSeries);

% Visualize results
figure;
subplot(2,1,1);
plot(timeSeries);
hold on;
for i = 1:length(foundCps)
    xline(foundCps(i), 'r--', 'LineWidth', 2);
end
title('Time Series with Detected Change Points');

subplot(2,1,2);
plot(clasp.Profiles);
title('ClaSP Classification Score Profile');
```

## Algorithm Details

The ClaSP algorithm works in the following steps:

1. **Subsequence Extraction**: Extract overlapping subsequences using a sliding window
2. **Normalization**: Apply z-normalization to each subsequence
3. **Distance Computation**: Calculate pairwise distances between subsequences
4. **k-NN Classification**: Use k-nearest neighbors to create binary classification problems
5. **Score Calculation**: Compute ROC-AUC scores for potential split points
6. **Profile Generation**: Create a profile of classification scores
7. **Change Point Detection**: Recursively find highest-scoring change points

## Requirements

- MATLAB R2016b or later
- Statistics and Machine Learning Toolbox (optional, for `pdist2` optimization)

## Performance Notes

- The algorithm complexity is O(n²m) where n is series length and m is window size
- For large time series (>10,000 points), consider using smaller window sizes or subsampling
- The Statistics Toolbox provides optimized distance calculations when available

## References

[1] Arik Ermshaus, Patrick Schäfer, and Ulf Leser. "ClaSP: parameter-free time series segmentation." Data Mining and Knowledge Discovery, 2023.

[2] Original Python implementation: https://github.com/aeon-toolkit/aeon

## License

This implementation follows the same license terms as the original aeon toolkit.