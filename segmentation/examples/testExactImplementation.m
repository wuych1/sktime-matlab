%% Test Exact ClaSP Implementation
% Test the exact ClaSP implementation and compare with the original

clear; clc; close all;
addpath(genpath('../'));

%% Test 1: Simple synthetic data
fprintf('=== Test 1: Simple Synthetic Data ===\n');

% Generate time series with clear change points
ts1 = sin(0.1 * (1:300)) + 0.1 * randn(1, 300);           % Regime 1
ts2 = 2 * sin(0.3 * (301:600)) + 0.2 * randn(1, 300);    % Regime 2
ts3 = 0.5 * cos(0.05 * (601:1000)) + 0.15 * randn(1, 400); % Regime 3

timeSeries = [ts1, ts2, ts3]';
trueChangePoints = [300, 600];

% Test exact implementation
claspExact = ClaSPSegmenterExact('PeriodLength', 30, 'NumChangePoints', 2);
tic;
foundCpsExact = claspExact.fitPredict(timeSeries);
timeExact = toc;

fprintf('Exact Implementation:\n');
fprintf('  Processing time: %.3f seconds\n', timeExact);
fprintf('  True change points: %s\n', mat2str(trueChangePoints));
fprintf('  Found change points: %s\n', mat2str(foundCpsExact'));

% Calculate accuracy
tolerance = 50;
accuracy = calculateDetectionAccuracy(trueChangePoints, foundCpsExact', tolerance);
fprintf('  Detection accuracy (±%d): %.2f%%\n', tolerance, accuracy * 100);

%% Test 2: Different parameters
fprintf('\n=== Test 2: Parameter Sensitivity ===\n');

windowSizes = [20, 30, 40];
for i = 1:length(windowSizes)
    fprintf('Testing window size %d:\n', windowSizes(i));

    claspTest = ClaSPSegmenterExact('PeriodLength', windowSizes(i), 'NumChangePoints', 2);
    foundCps = claspTest.fitPredict(timeSeries);
    accuracy = calculateDetectionAccuracy(trueChangePoints, foundCps', tolerance);

    fprintf('  Found: %s, Accuracy: %.2f%%\n', mat2str(foundCps'), accuracy * 100);
end

%% Test 3: API compatibility
fprintf('\n=== Test 3: API Compatibility Test ===\n');

% Test the Python-like API
periodSize = 30;
clasp = ClaSPSegmenterExact('PeriodLength', periodSize, 'NumChangePoints', 2);

% Fit and predict
foundCps = clasp.fitPredict(timeSeries);

% Access results
profiles = clasp.Profiles;
scores = clasp.Scores;

fprintf('The found change points are: %s\n', mat2str(foundCps'));
fprintf('Number of profiles: %d\n', length(profiles));
fprintf('Scores: %s\n', mat2str(scores, 3));

%% Visualization
figure('Position', [100, 100, 1200, 600]);

subplot(2,1,1);
plot(timeSeries, 'b-', 'LineWidth', 1.5);
hold on;
for i = 1:length(trueChangePoints)
    xline(trueChangePoints(i), 'g--', 'LineWidth', 2);
end
for i = 1:length(foundCpsExact)
    xline(foundCpsExact(i), 'r-', 'LineWidth', 2);
end
title('Time Series with Change Points (Exact Implementation)');
xlabel('Time');
ylabel('Value');
legend('Time Series', 'True Change Points', 'Detected Change Points', 'Location', 'best');
grid on;

subplot(2,1,2);
if ~isempty(profiles) && length(profiles) > 0
    plot(profiles{1}, 'k-', 'LineWidth', 1.5);
    title('ClaSP Profile (Global)');
    xlabel('Time');
    ylabel('Classification Score');
    grid on;
end

%% Helper function
function accuracy = calculateDetectionAccuracy(trueCP, detectedCP, tolerance)
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

fprintf('\n=== Testing completed! ===\n');