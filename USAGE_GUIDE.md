# ClaSP MATLAB - Comprehensive Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Parameter Selection](#parameter-selection)
3. [Common Use Cases](#common-use-cases)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)
6. [Comparison with Python](#comparison-with-python)

## Quick Start

### Basic Usage
```matlab
% Load your time series data
data = readmatrix('your_data.csv');

% Create ClaSP segmenter with default settings
clasp = ClaSPSegmenter('PeriodLength', 20, 'NumChangePoints', 3);

% Find change points
changePoints = clasp.fitPredict(data);

% Access additional information
scores = clasp.Scores;     % Confidence scores for each change point
profiles = clasp.Profiles; % Classification profiles for each segment
```

## Parameter Selection

### PeriodLength (Window Size)
The most critical parameter. Choose based on:
- **Pattern length**: Set to approximately half the length of patterns you expect
- **Data frequency**: Higher frequency data may need smaller windows
- **Computation time**: Smaller windows are faster but may miss longer patterns

```matlab
% For hourly data with daily patterns
windowSize = 12;  % Half of 24-hour period

% For high-frequency sensor data
windowSize = 10;  % Small window for rapid changes

% For monthly economic data
windowSize = 6;   % Half-year patterns
```

### NumChangePoints
Set based on domain knowledge or use iterative approach:

```matlab
% Try different numbers and compare results
for nCPs = 1:5
    clasp = ClaSPSegmenter('PeriodLength', 20, 'NumChangePoints', nCPs);
    cps = clasp.fitPredict(data);
    fprintf('With %d CPs: %s\n', nCPs, mat2str(cps'));
end
```

### ExclusionRadius
Controls sensitivity and boundary handling:

| Value | Use Case | Effect |
|-------|----------|--------|
| 0.10 | Default, robust | Excludes 10% around each CP and boundaries |
| 0.05 | Balanced | Good for most applications |
| 0.02 | Sensitive | Allows boundary detections |
| 0.01 | Very sensitive | May detect spurious changes |

```matlab
% For detecting boundary changes
clasp = ClaSPSegmenter('PeriodLength', 20, ...
                        'NumChangePoints', 3, ...
                        'ExclusionRadius', 0.02);
```

## Common Use Cases

### 1. Financial Time Series
```matlab
% Stock price regime detection
stockPrices = readmatrix('stock_prices.csv');

% Use smaller window for intraday patterns
clasp = ClaSPSegmenter('PeriodLength', 30, ...
                        'NumChangePoints', 5, ...
                        'ExclusionRadius', 0.05);
changePoints = clasp.fitPredict(stockPrices);

% Visualize regimes
figure;
plot(stockPrices);
hold on;
for cp = changePoints'
    xline(cp, 'r--', 'LineWidth', 2);
end
title('Stock Price Regimes');
```

### 2. Sensor Data Anomaly Detection
```matlab
% IoT sensor readings
sensorData = readmatrix('sensor_readings.csv');

% Detect anomalies as change points
clasp = ClaSPSegmenter('PeriodLength', 10, ...
                        'NumChangePoints', 10, ...  % More CPs for anomalies
                        'ExclusionRadius', 0.03);    % Tighter exclusion

anomalies = clasp.fitPredict(sensorData);
scores = clasp.Scores;

% Filter by score threshold
significantAnomalies = anomalies(scores > 0.5);
```

### 3. Climate Data Analysis
```matlab
% Temperature time series
temperature = readmatrix('temperature_data.csv');

% Detect seasonal transitions
clasp = ClaSPSegmenter('PeriodLength', 90, ...  % Quarterly patterns
                        'NumChangePoints', 4, ...  % Seasonal changes
                        'ExclusionRadius', 0.1);

seasons = clasp.fitPredict(temperature);
```

### 4. Manufacturing Process Control
```matlab
% Production metrics
production = readmatrix('production_metrics.csv');

% Detect process shifts
clasp = ClaSPSegmenter('PeriodLength', 20, ...
                        'NumChangePoints', 5, ...
                        'ExclusionRadius', 0.05);

processShifts = clasp.fitPredict(production);

% Alert if recent shift detected
if any(processShifts > length(production) - 100)
    warning('Recent process shift detected!');
end
```

## Performance Optimization

### For Large Datasets (>10,000 points)

1. **Use smaller windows**:
```matlab
% Faster computation with smaller window
clasp = ClaSPSegmenter('PeriodLength', 10, ...  % Small window
                        'NumChangePoints', 5);
```

2. **Process in chunks**:
```matlab
% Split long series into overlapping chunks
chunkSize = 5000;
overlap = 500;

for i = 1:chunkSize-overlap:length(data)-chunkSize
    chunk = data(i:i+chunkSize-1);
    localCPs = clasp.fitPredict(chunk);
    globalCPs = i + localCPs - 1;  % Convert to global indices
end
```

3. **Reduce exclusion radius**:
```matlab
% Smaller exclusion = less computation in some cases
clasp = ClaSPSegmenter('PeriodLength', 15, ...
                        'NumChangePoints', 3, ...
                        'ExclusionRadius', 0.03);
```

### Memory Management
```matlab
% Clear unnecessary variables
clear profiles knnMask;  % After getting change points

% Process incrementally
clasp = ClaSPSegmenter('PeriodLength', 20, 'NumChangePoints', 1);
for i = 1:numSegments
    segment = getSegment(data, i);
    cp = clasp.fitPredict(segment);
    % Process each change point immediately
    processChangePoint(cp);
end
```

## Troubleshooting

### Issue: No change points detected
**Solutions**:
- Reduce ExclusionRadius
- Check if data has sufficient variation
- Try different window sizes

```matlab
% Diagnostic code
if isempty(changePoints)
    fprintf('No CPs found. Data statistics:\n');
    fprintf('  Std: %.4f\n', std(data));
    fprintf('  Range: %.4f\n', range(data));

    % Try with lower exclusion
    clasp.ExclusionRadius = 0.02;
    changePoints = clasp.fitPredict(data);
end
```

### Issue: Change points only at boundaries
**Cause**: Boundary effects or artificial extrema
**Solutions**:
```matlab
% Filter boundary detections
n = length(data);
boundaryThreshold = 0.1 * n;  % 10% from edges

validCPs = changePoints(changePoints > boundaryThreshold & ...
                        changePoints < n - boundaryThreshold);
```

### Issue: Too many trivial change points
**Solutions**:
```matlab
% Increase exclusion radius
clasp.ExclusionRadius = 0.15;  % Larger exclusion zones

% Or filter by score
minScore = median(clasp.Scores);
significantCPs = changePoints(clasp.Scores > minScore);
```

### Issue: Computation taking too long
**Solutions**:
```matlab
% 1. Reduce data resolution
dataReduced = data(1:10:end);  % Every 10th point

% 2. Use profile directly for quick analysis
transformer = ClaSPTransformer();
profile = transformer.transform(data, 20);
[~, quickCP] = max(profile);  % Single most likely change point
```

## Comparison with Python

### Equivalence Testing
```matlab
% Save MATLAB results
matlabCPs = clasp.fitPredict(data);
writematrix(matlabCPs, 'matlab_cps.csv');

% In Python:
% from sktime.detection.clasp import ClaSPSegmentation
% clasp = ClaSPSegmentation(period_length=20, n_cps=3)
% python_cps = clasp.fit_predict(data)

% Compare results
pythonCPs = readmatrix('python_cps.csv');
fprintf('MATLAB CPs: %s\n', mat2str(matlabCPs'));
fprintf('Python CPs: %s\n', mat2str(pythonCPs'));
```

### Key Differences
1. **Boundary handling**: Both exclude boundaries, but exact positions may vary
2. **Numerical precision**: Minor differences expected due to floating-point computation
3. **Performance**: MATLAB may be slower for very large datasets without optimization

### Ensuring Compatibility
```matlab
% Match Python exactly with same parameters
clasp = ClaSPSegmenter();
clasp.PeriodLength = 10;        % period_length in Python
clasp.NumChangePoints = 2;       % n_cps in Python
clasp.ExclusionRadius = 0.05;    % exclusion_radius in Python

changePoints = clasp.fitPredict(data);
```

## Advanced Features

### Custom Profile Analysis
```matlab
% Get the raw profile for custom analysis
transformer = ClaSPTransformer();
[profile, knnMask] = transformer.transform(data, 20);

% Find all peaks in profile
[peaks, locations] = findpeaks(profile, 'MinPeakHeight', 0.3);

% Custom change point selection
customCPs = locations(peaks > quantile(peaks, 0.75));
```

### Iterative Refinement
```matlab
% Start with coarse detection
coarseClasp = ClaSPSegmenter('PeriodLength', 50, 'NumChangePoints', 2);
coarseCPs = coarseClasp.fitPredict(data);

% Refine each segment
for i = 1:length(coarseCPs)+1
    if i == 1
        segment = data(1:coarseCPs(1));
    elseif i == length(coarseCPs)+1
        segment = data(coarseCPs(end):end);
    else
        segment = data(coarseCPs(i-1):coarseCPs(i));
    end

    % Fine-grained detection in segment
    fineClasp = ClaSPSegmenter('PeriodLength', 10, 'NumChangePoints', 1);
    localCP = fineClasp.fitPredict(segment);
    % Convert to global index and store
end
```

## Best Practices

1. **Always visualize results**:
```matlab
plotClaSPResults(data, changePoints, clasp.Profiles{1});
```

2. **Validate with domain knowledge**:
```matlab
% Check if detected changes align with known events
knownEvents = [100, 450, 780];
for event = knownEvents
    nearestCP = changePoints(abs(changePoints - event) == ...
                             min(abs(changePoints - event)));
    fprintf('Event at %d mapped to CP at %d\n', event, nearestCP);
end
```

3. **Use ensemble approach for robustness**:
```matlab
% Try multiple window sizes
windows = [10, 20, 30, 40];
allCPs = [];
for w = windows
    clasp = ClaSPSegmenter('PeriodLength', w, 'NumChangePoints', 3);
    cps = clasp.fitPredict(data);
    allCPs = [allCPs; cps];
end
% Cluster or vote on final change points
finalCPs = clusterChangePoints(allCPs);
```

## References

For more details on the algorithm:
- Schäfer, P., Ermshaus, A., & Leser, U. (2021). ClaSP - Time Series Segmentation. CIKM.
- [Python implementation documentation](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.detection.clasp.ClaSPSegmentation.html)