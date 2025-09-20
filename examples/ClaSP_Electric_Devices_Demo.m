%% ClaSP Electric Devices Demo - Subset for Fast Demonstration
% This demo uses a subset of the electric devices data for quick testing

%% Setup
clear; clc; close all;
addpath('../segmentation');
addpath('../testing/data');

%% Load Full Dataset
fprintf('Loading Electric Devices dataset...\n');
ts_full = readmatrix('electric_devices_timeseries.csv');
true_cps_full = readmatrix('electric_devices_changepoints.csv')';
period_size = readmatrix('electric_devices_period.txt');

fprintf('Full dataset: %d samples\n', length(ts_full));
fprintf('Full true CPs: [%s]\n', num2str(true_cps_full'));

%% Extract Subset for Demo (first 2000 samples)
subset_end = 2000;
ts = ts_full(1:subset_end);
true_cps = true_cps_full(true_cps_full <= subset_end);

fprintf('\nDemo subset: %d samples\n', length(ts));
fprintf('Subset true CPs: [%s]\n', num2str(true_cps'));

%% Visualize Subset
figure('Position', [100, 100, 1200, 400]);
plot(ts, 'b-', 'LineWidth', 0.8);
hold on;
for cp = true_cps'
    xline(cp, 'g:', 'LineWidth', 2);
end
xlabel('Time');
ylabel('Energy Consumption');
title(sprintf('Electric Devices Subset (first %d samples)', subset_end));
grid on;

%% Run ClaSP on Subset
fprintf('\nRunning ClaSP segmentation on subset...\n');

clasp = ClaSPSegmenter('PeriodLength', period_size, ...
                       'NumChangePoints', length(true_cps), ...
                       'ExclusionRadius', 0.05);

tic;
detected_cps = clasp.fitPredict(ts);
elapsed = toc;

detected_cps = sort(detected_cps);

%% Results
fprintf('\nRESULTS FOR SUBSET:\n');
fprintf('===================\n');
fprintf('Processing time: %.1f seconds\n', elapsed);
fprintf('True CPs:     [%s]\n', num2str(true_cps'));
fprintf('Detected CPs: [%s]\n', num2str(detected_cps'));

% Calculate accuracy
if ~isempty(true_cps) && ~isempty(detected_cps)
    errors = zeros(length(true_cps), 1);
    for i = 1:min(length(true_cps), length(detected_cps))
        errors(i) = abs(true_cps(i) - detected_cps(i));
        fprintf('CP%d error: %d samples (%.1f%% of subset)\n', ...
                i, errors(i), 100*errors(i)/length(ts));
    end

    tolerance = 0.05 * length(ts);
    accurate = sum(errors < tolerance);
    accuracy = 100 * accurate / length(true_cps);
    fprintf('Accuracy (5%% tolerance): %.1f%%\n', accuracy);
else
    fprintf('No change points found or expected in subset\n');
end

%% Visualization
figure('Position', [100, 100, 1200, 600]);

% Original with change points
subplot(2,1,1);
plot(ts, 'b-', 'LineWidth', 0.8);
hold on;
for cp = true_cps'
    xline(cp, 'g:', 'LineWidth', 2);
end
for cp = detected_cps'
    xline(cp, 'r--', 'LineWidth', 2);
end
xlabel('Time');
ylabel('Value');
title('ClaSP Results: Green=True, Red=Detected');
legend('Time Series', 'True CPs', 'Detected CPs', 'Location', 'best');
grid on;

% Score profile if available
subplot(2,1,2);
if ~isempty(clasp.Profiles)
    profile = clasp.Profiles{1};
    plot(profile, 'k-', 'LineWidth', 1);
    hold on;

    % Mark detected change points on profile
    for cp = detected_cps'
        if cp <= length(profile)
            plot(cp, profile(cp), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
    end

    % Mark true change points for comparison
    for cp = true_cps'
        if cp <= length(profile)
            xline(cp, 'g:', 'LineWidth', 1);
        end
    end

    xlabel('Time');
    ylabel('Classification Score');
    title('ClaSP Score Profile (peaks indicate good change points)');
    grid on;
else
    text(0.5, 0.5, 'No profile available', 'HorizontalAlignment', 'center');
end

%% Compare with Python Results (if available)
try
    python_cps = readmatrix('electric_devices_python_cps.csv')';
    python_subset = python_cps(python_cps <= subset_end);

    fprintf('\nComparison with Python sktime:\n');
    fprintf('Python CPs (subset): [%s]\n', num2str(python_subset'));
    fprintf('MATLAB CPs (subset): [%s]\n', num2str(detected_cps'));

    if ~isempty(python_subset) && ~isempty(detected_cps)
        matlab_python_diff = abs(detected_cps(1:min(end,length(python_subset))) - ...
                                python_subset(1:min(end,length(detected_cps))));
        fprintf('MATLAB vs Python differences: [%s]\n', num2str(matlab_python_diff'));
    end
catch
    fprintf('\nPython comparison data not available\n');
end

%% Summary
fprintf('\n=== DEMO SUMMARY ===\n');
fprintf('This demo shows ClaSP working on a subset of the Electric Devices dataset.\n');
fprintf('For the full dataset analysis, use ClaSP_Electric_Devices_Example.m\n');
fprintf('(Note: Full dataset processing takes several minutes)\n');

%% Instructions for Full Analysis
fprintf('\nTo run the complete analysis:\n');
fprintf('1. Use ClaSP_Electric_Devices_Example.m for comprehensive analysis\n');
fprintf('2. Processing time: ~5-10 minutes for full dataset\n');
fprintf('3. All visualizations and parameter analysis included\n\n');