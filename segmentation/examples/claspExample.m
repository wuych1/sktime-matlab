%% ClaSP Time Series Segmentation Example
% This script demonstrates how to use the ClaSPSegmenterExact class for
% time series segmentation. It includes examples with synthetic data
% and shows how to visualize the results.

clear; clc; close all;

%% Add segmentation folder to path
addpath(genpath('../'));

%% Example 1: Simple Synthetic Time Series with Known Change Points
fprintf('=== Example 1: Synthetic Time Series ===\n');

% Generate synthetic time series with clear regime changes
n = 1000;
t = 1:n;

% Create time series with 3 distinct regimes
ts1 = sin(0.1 * t(1:300)) + 0.1 * randn(1, 300);           % Regime 1: Low frequency sine
ts2 = 2 * sin(0.3 * t(301:600)) + 0.2 * randn(1, 300);    % Regime 2: Higher amplitude and frequency
ts3 = 0.5 * cos(0.05 * t(601:1000)) + 0.15 * randn(1, 400); % Regime 3: Cosine with low frequency

timeSeries = [ts1, ts2, ts3]';
trueChangePoints = [300, 600];  % Known change points

% Create ClaSP segmenter
clasp = ClaSPSegmenterExact('PeriodLength', 50, 'NumChangePoints', 2, 'ExclusionRadius', 0.05);

% Fit and predict change points
fprintf('Fitting ClaSP model...\n');
tic;
foundCps = clasp.fitPredict(timeSeries);
elapsedTime = toc;

% Display results
fprintf('Processing time: %.3f seconds\n', elapsedTime);
fprintf('True change points: %s\n', mat2str(trueChangePoints));
fprintf('Found change points: %s\n', mat2str(foundCps'));

% Calculate detection accuracy
accuracy = calculateDetectionAccuracy(trueChangePoints, foundCps', 50);
fprintf('Detection accuracy: %.2f%%\n', accuracy * 100);

% Visualize results
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Original time series with change points
subplot(3, 1, 1);
plot(timeSeries, 'b-', 'LineWidth', 1.5);
hold on;
for i = 1:length(trueChangePoints)
    xline(trueChangePoints(i), 'g--', 'LineWidth', 2, 'Label', 'True CP');
end
for i = 1:length(foundCps)
    xline(foundCps(i), 'r-', 'LineWidth', 2, 'Label', 'Found CP');
end
title('Time Series with Change Points');
xlabel('Time');
ylabel('Value');
legend('Time Series', 'True Change Points', 'Detected Change Points', 'Location', 'best');
grid on;

% Plot 2: ClaSP profiles
subplot(3, 1, 2);
if ~isempty(clasp.Profiles) && length(clasp.Profiles) > 0
    plot(clasp.Profiles{1}, 'k-', 'LineWidth', 1.5);
    title('ClaSP Classification Score Profile');
    xlabel('Time');
    ylabel('Classification Score');
    grid on;
end

% Plot 3: Change point scores
subplot(3, 1, 3);
if ~isempty(clasp.Scores)
    stem(foundCps, clasp.Scores, 'ro', 'LineWidth', 2, 'MarkerSize', 8);
    title('Change Point Scores');
    xlabel('Time');
    ylabel('Score');
    grid on;
end

%% Example 2: Real-world-like Data with Multiple Regimes
fprintf('\n=== Example 2: Multi-regime Time Series ===\n');

% Generate more complex time series
n = 800;
t = 1:n;

% Multiple regimes with different characteristics
regime1 = cumsum(0.02 * randn(1, 200)) + sin(0.1 * t(1:200));      % Random walk + sine
regime2 = 3 + 0.5 * sin(0.5 * t(201:400)) + 0.1 * randn(1, 200);  % Shifted sine
regime3 = 2 * exp(-0.01 * (t(401:600) - 400)) + 0.2 * randn(1, 200); % Exponential decay
regime4 = 1 + 0.3 * (t(601:800) - 600) / 200 + 0.15 * randn(1, 200); % Linear trend

complexTs = [regime1, regime2, regime3, regime4]';

% Use ClaSP with automatic window size detection
claspAuto = ClaSPSegmenterExact('NumChangePoints', 5);

% Find optimal window size
optimalWindow = claspAuto.findDominantWindowSizes(complexTs);
fprintf('Optimal window size: %d\n', optimalWindow);

% Set the optimal window size and segment
claspAuto.PeriodLength = optimalWindow;
foundCpsAuto = claspAuto.fitPredict(complexTs);

fprintf('Found change points: %s\n', mat2str(foundCpsAuto'));

% Visualize complex example
figure('Position', [150, 150, 1200, 600]);

subplot(2, 1, 1);
plot(complexTs, 'b-', 'LineWidth', 1.5);
hold on;
for i = 1:length(foundCpsAuto)
    xline(foundCpsAuto(i), 'r-', 'LineWidth', 2);
end
title('Complex Multi-regime Time Series');
xlabel('Time');
ylabel('Value');
legend('Time Series', 'Detected Change Points', 'Location', 'best');
grid on;

subplot(2, 1, 2);
if ~isempty(claspAuto.Profiles) && length(claspAuto.Profiles) > 0
    plot(claspAuto.Profiles{1}, 'k-', 'LineWidth', 1.5);
    title('ClaSP Profile for Complex Time Series');
    xlabel('Time');
    ylabel('Classification Score');
    grid on;
end

%% Example 3: Parameter Sensitivity Analysis
fprintf('\n=== Example 3: Parameter Sensitivity ===\n');

% Test different parameter settings
windowSizes = [20, 40, 60, 80];
exclusionRadii = [0.01, 0.05, 0.1, 0.2];

% Use the simple synthetic data from Example 1
testResults = cell(length(windowSizes), length(exclusionRadii));

for i = 1:length(windowSizes)
    for j = 1:length(exclusionRadii)
        clasp_test = ClaSPSegmenterExact('PeriodLength', windowSizes(i), ...
                                   'NumChangePoints', 2, ...
                                   'ExclusionRadius', exclusionRadii(j));

        foundCps_test = clasp_test.fitPredict(timeSeries);
        accuracy_test = calculateDetectionAccuracy(trueChangePoints, foundCps_test', 50);

        testResults{i, j} = struct('WindowSize', windowSizes(i), ...
                                  'ExclusionRadius', exclusionRadii(j), ...
                                  'ChangePoints', foundCps_test', ...
                                  'Accuracy', accuracy_test);

        fprintf('Window: %2d, Exclusion: %.2f, Accuracy: %.2f%%\n', ...
               windowSizes(i), exclusionRadii(j), accuracy_test * 100);
    end
end

%% Example 4: API Usage Demonstration
fprintf('\n=== Example 4: API Usage Pattern ===\n');

% Demonstrate the Python-like API
periodSize = 30;
clasp = ClaSPSegmenterExact('PeriodLength', periodSize, 'NumChangePoints', 5);

% Fit and predict (matching Python API style)
foundCps = clasp.fitPredict(timeSeries);

% Access results (matching Python API style)
profiles = clasp.Profiles;
scores = clasp.Scores;

fprintf('The found change points are: %s\n', mat2str(foundCps'));
fprintf('Number of profiles computed: %d\n', size(profiles, 2));
fprintf('Change point scores: %s\n', mat2str(scores, 3));

%% Helper Functions

function accuracy = calculateDetectionAccuracy(trueCP, detectedCP, tolerance)
    % Calculate detection accuracy with tolerance
    if isempty(detectedCP)
        accuracy = 0;
        return;
    end

    correct = 0;
    for i = 1:length(trueCP)
        distances = abs(detectedCP - trueCP(i));
        if any(distances <= tolerance)
            correct = correct + 1;
        end
    end

    accuracy = correct / length(trueCP);
end

fprintf('\n=== Examples completed successfully! ===\n');