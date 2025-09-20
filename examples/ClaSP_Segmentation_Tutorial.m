%% ClaSP Time Series Segmentation Tutorial
% This tutorial demonstrates how to use the ClaSP (Classification Score Profile)
% algorithm for time series segmentation in MATLAB, similar to the sktime Python
% implementation.

%% Introduction
% ClaSP is a novel time series segmentation algorithm that finds change points
% by evaluating how well a classifier can distinguish between subsequences on
% either side of a potential split point. The algorithm:
%
% * Hierarchically splits the time series
% * Uses k-NN classification to score each potential split
% * Selects split points with highest classification accuracy (ROC-AUC)
% * Automatically excludes boundary regions to avoid spurious detections

%% Setup
% First, let's add the necessary paths and clear the workspace

clear; clc; close all;

% Add path to the segmentation module
addpath('../segmentation');
addpath('../testing');  % For test data

% Set random seed for reproducibility
rng(42);

%% Example 1: Simple Step Function
% Let's start with a basic example - a time series with a clear step change

% Generate synthetic data with a step change
n = 200;
t = 1:n;
ts_step = [zeros(1, 80) + randn(1, 80)*0.1, ...
           ones(1, 120) + randn(1, 120)*0.1];

% Create ClaSP segmenter with appropriate window size
clasp_step = ClaSPSegmenter('PeriodLength', 20, ...
                            'NumChangePoints', 1, ...
                            'ExclusionRadius', 0.05);

% Find change points
cp_step = clasp_step.fitPredict(ts_step');

% Visualize results
figure('Position', [100, 100, 800, 400]);
plot(t, ts_step, 'b-', 'LineWidth', 1.5);
hold on;
xline(cp_step, 'r--', 'LineWidth', 2);
xline(80, 'g:', 'LineWidth', 2);  % True change point
xlabel('Time');
ylabel('Value');
title('Simple Step Function Segmentation');
legend('Time Series', 'Detected Change Point', 'True Change Point', ...
       'Location', 'best');
grid on;

fprintf('True change point: %d\n', 80);
fprintf('Detected change point: %d\n', cp_step);
fprintf('Error: %d samples\n', abs(cp_step - 80));

%% Example 2: Multiple Level Changes
% Now let's try a more complex example with multiple change points

% Generate data with three distinct levels
n = 300;
t = 1:n;
ts_multilevel = [ones(1, 100)*0 + randn(1, 100)*0.1, ...
                 ones(1, 100)*5 + randn(1, 100)*0.1, ...
                 ones(1, 100)*2 + randn(1, 100)*0.1];

% Configure ClaSP for multiple change points
clasp_multi = ClaSPSegmenter('PeriodLength', 20, ...
                             'NumChangePoints', 2, ...
                             'ExclusionRadius', 0.05);

% Detect change points
cp_multi = clasp_multi.fitPredict(ts_multilevel');

% Visualize
figure('Position', [100, 100, 800, 400]);
plot(t, ts_multilevel, 'b-', 'LineWidth', 1.5);
hold on;
for cp = cp_multi'
    xline(cp, 'r--', 'LineWidth', 2);
end
xline(100, 'g:', 'LineWidth', 2);  % True change points
xline(200, 'g:', 'LineWidth', 2);
xlabel('Time');
ylabel('Value');
title('Multiple Level Changes Segmentation');
legend('Time Series', 'Detected Change Points', 'True Change Points', ...
       'Location', 'best');
grid on;

fprintf('\nTrue change points: [100, 200]\n');
fprintf('Detected change points: [%s]\n', num2str(cp_multi'));

%% Example 3: Frequency Change in Sinusoidal Signal
% ClaSP can also detect changes in periodic patterns

% Generate sinusoid with frequency change
n = 400;
t = linspace(0, 40, n);
ts_freq = [sin(2*pi*1*t(1:200)), ...  % 1 Hz
           sin(2*pi*3*t(201:400))];   % 3 Hz
ts_freq = ts_freq + randn(1, n)*0.05;  % Add noise

% Use larger window to capture the periodicity
clasp_freq = ClaSPSegmenter('PeriodLength', 30, ...
                            'NumChangePoints', 1, ...
                            'ExclusionRadius', 0.05);

cp_freq = clasp_freq.fitPredict(ts_freq');

% Visualize
figure('Position', [100, 100, 800, 400]);
plot(t, ts_freq, 'b-', 'LineWidth', 1);
hold on;
xline(t(cp_freq), 'r--', 'LineWidth', 2);
xline(t(200), 'g:', 'LineWidth', 2);  % True change point
xlabel('Time');
ylabel('Value');
title('Frequency Change Detection in Sinusoidal Signal');
legend('Time Series', 'Detected Change Point', 'True Change Point', ...
       'Location', 'best');
grid on;

fprintf('\nTrue frequency change at index: 200\n');
fprintf('Detected change point: %d\n', cp_freq);

%% Example 4: Using Electric Devices Dataset
% Load and segment real-world electric devices data

% Load the electric devices test data
try
    load('../testing/electric_devices_test.mat', 'X');

    % Use the first time series
    ts_electric = X(1, :);
    n_electric = length(ts_electric);

    % Find optimal window size using dominant frequency
    clasp_finder = ClaSPSegmenter();
    optimal_window = clasp_finder.findDominantWindowSizes(ts_electric');
    fprintf('\nOptimal window size found: %d\n', optimal_window);

    % Segment with optimal parameters
    clasp_electric = ClaSPSegmenter('PeriodLength', optimal_window, ...
                                    'NumChangePoints', 4, ...
                                    'ExclusionRadius', 0.05);

    cp_electric = clasp_electric.fitPredict(ts_electric');

    % Visualize
    figure('Position', [100, 100, 900, 500]);
    subplot(2,1,1);
    plot(ts_electric, 'b-', 'LineWidth', 1);
    hold on;
    for cp = cp_electric'
        xline(cp, 'r--', 'LineWidth', 2);
    end
    xlabel('Time');
    ylabel('Value');
    title('Electric Devices Time Series Segmentation');
    legend('Time Series', 'Detected Change Points', 'Location', 'best');
    grid on;

    % Plot the classification score profile
    subplot(2,1,2);
    if ~isempty(clasp_electric.Profiles)
        profile = clasp_electric.Profiles{1};
        plot(profile, 'k-', 'LineWidth', 1.5);
        hold on;
        plot(cp_electric(1), profile(cp_electric(1)), 'ro', ...
             'MarkerSize', 8, 'MarkerFaceColor', 'r');
        xlabel('Time');
        ylabel('Classification Score');
        title('ClaSP Score Profile');
        grid on;
    end

    fprintf('Detected %d change points in electric devices data\n', ...
            length(cp_electric));
    fprintf('Change points at indices: [%s]\n', num2str(cp_electric'));

catch ME
    fprintf('Note: Electric devices dataset not found. Skipping this example.\n');
    fprintf('Error: %s\n', ME.message);
end

%% Parameter Tuning Guide
% The key parameters for ClaSP segmentation are:

% 1. PeriodLength (window_size)
%    - Should capture the characteristic pattern in your data
%    - Too small: windows become too similar
%    - Too large: windows may span multiple segments
%    - Use findDominantWindowSizes() for automatic selection

% 2. NumChangePoints (n_cps)
%    - Number of change points to detect
%    - Algorithm stops early if no good splits are found

% 3. ExclusionRadius
%    - Fraction of series length to exclude around boundaries
%    - Default 0.05 (5%) works well for most cases
%    - Reduce for detecting near-boundary changes

%% Example 5: Parameter Sensitivity Analysis
% Let's examine how window size affects detection

ts_param = [sin(2*pi*t(1:150)/20), sin(2*pi*t(151:300)/10)] + ...
           randn(1, 300)*0.05;

window_sizes = [10, 20, 30, 40];
figure('Position', [100, 100, 900, 600]);

for i = 1:length(window_sizes)
    ws = window_sizes(i);
    clasp_param = ClaSPSegmenter('PeriodLength', ws, ...
                                 'NumChangePoints', 1);
    cp = clasp_param.fitPredict(ts_param');

    subplot(2, 2, i);
    plot(ts_param, 'b-');
    hold on;
    xline(cp, 'r--', 'LineWidth', 2);
    xline(150, 'g:', 'LineWidth', 2);
    title(sprintf('Window Size = %d', ws));
    xlabel('Time');
    ylabel('Value');
    grid on;

    fprintf('Window size %d: Detected CP at %d (Error: %d)\n', ...
            ws, cp, abs(cp - 150));
end
sgtitle('Effect of Window Size on Change Point Detection');

%% Example 6: Visualizing the ClaSP Profile
% Understanding the classification score profile

% Generate data with clear change
n = 200;
ts_profile = [ones(1, 100)*2, ones(1, 100)*5] + randn(1, n)*0.1;

% Create transformer directly to get the profile
transformer = ClaSPTransformer();
[profile, ~] = transformer.transform(ts_profile', 20, ...
                                     'ExclusionRadius', 0.05);

% Visualize time series and profile together
figure('Position', [100, 100, 800, 600]);

subplot(2,1,1);
plot(ts_profile, 'b-', 'LineWidth', 1.5);
xline(100, 'g:', 'LineWidth', 2);
xlabel('Time');
ylabel('Value');
title('Time Series with Step Change');
grid on;

subplot(2,1,2);
plot(profile, 'k-', 'LineWidth', 1.5);
hold on;
[max_score, max_idx] = max(profile);
plot(max_idx, max_score, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('Split Point');
ylabel('ROC-AUC Score');
title('ClaSP Classification Score Profile');
grid on;

fprintf('\nProfile peak at index %d with score %.3f\n', max_idx, max_score);

%% Summary
% ClaSP is a powerful algorithm for time series segmentation that:
%
% * Automatically identifies semantically meaningful change points
% * Works across various domains and signal types
% * Provides interpretable classification scores
% * Handles multiple change points hierarchically
% * Includes boundary exclusion to prevent spurious detections
%
% Key tips for using ClaSP:
%
% * Start with findDominantWindowSizes() for automatic parameter selection
% * Adjust ExclusionRadius if you need to detect near-boundary changes
% * Examine the score profile to understand the algorithm's decisions
% * Consider the trade-off between window size and detection accuracy

fprintf('\n=== Tutorial Complete ===\n');
fprintf('ClaSP successfully demonstrated on various segmentation tasks.\n');