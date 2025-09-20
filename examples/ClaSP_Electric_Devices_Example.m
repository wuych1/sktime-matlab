%% Time Series Segmentation with ClaSP - Electric Devices Dataset
% This example demonstrates ClaSP segmentation on the Electric Devices dataset,
% following the exact structure of the sktime Python tutorial.
%
% Dataset: Electric Devices - energy consumption profiles of household devices
% The data contains 4 known change points where the operational state changes.

%% 1. Setup and Data Loading
clear; clc; close all;

% Add necessary paths
addpath('../segmentation');
addpath('../testing/data');

fprintf('========================================\n');
fprintf('   ClaSP Segmentation - Electric Devices\n');
fprintf('========================================\n\n');

%% 2. Load the Electric Devices Dataset
fprintf('Loading Electric Devices dataset...\n');

% Load time series data
ts = readmatrix('electric_devices_timeseries.csv');
n = length(ts);

% Load ground truth change points
true_cps = readmatrix('electric_devices_changepoints.csv')';
true_cps = true_cps(:);  % Ensure column vector

% Load the optimal period size
period_size = readmatrix('electric_devices_period.txt');

fprintf('Dataset information:\n');
fprintf('  - Time series length: %d\n', n);
fprintf('  - True change points: [%s]\n', num2str(true_cps'));
fprintf('  - Number of change points: %d\n', length(true_cps));
fprintf('  - Recommended period size: %d\n\n', period_size);

%% 3. Visualize the Raw Time Series
figure('Position', [100, 100, 1200, 400]);
plot(ts, 'b-', 'LineWidth', 0.8);
hold on;

% Add vertical lines for true change points
for i = 1:length(true_cps)
    xline(true_cps(i), 'g:', 'LineWidth', 1.5, ...
          'Label', sprintf('CP%d', i), 'LabelOrientation', 'horizontal');
end

xlabel('Time');
ylabel('Energy Consumption');
title('Electric Devices Time Series with True Change Points');
grid on;
legend('Time Series', 'True Change Points', 'Location', 'best');

%% 4. Understanding ClaSP: The Classification Score Profile
fprintf('Computing ClaSP profile to understand the algorithm...\n');

% Create transformer to compute the profile
transformer = ClaSPTransformer();

% Compute the classification score profile
[profile, ~] = transformer.transform(ts, period_size, ...
                                     'K', 3, ...
                                     'ExclusionRadius', 0.05);

% Visualize the profile
figure('Position', [100, 100, 1200, 600]);

subplot(2,1,1);
plot(ts, 'b-', 'LineWidth', 0.8);
hold on;
for i = 1:length(true_cps)
    xline(true_cps(i), 'g:', 'LineWidth', 1.5);
end
xlabel('Time');
ylabel('Value');
title('Electric Devices Time Series');
grid on;

subplot(2,1,2);
plot(profile, 'k-', 'LineWidth', 1);
hold on;

% Find and mark the peaks in the profile
[peaks, peak_locs] = findpeaks(profile, 'MinPeakHeight', 0.1, 'MinPeakDistance', 100);
plot(peak_locs, peaks, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

for i = 1:length(true_cps)
    xline(true_cps(i), 'g:', 'LineWidth', 1.5);
end

xlabel('Time');
ylabel('Classification Score (ROC-AUC)');
title('ClaSP Score Profile - Higher values indicate better split points');
grid on;
legend('Score Profile', 'Detected Peaks', 'True CPs', 'Location', 'best');

fprintf('Profile computed. Peaks indicate potential change points.\n\n');

%% 5. Automatic Window Size Selection
fprintf('Finding optimal window size using dominant frequency...\n');

% Use ClaSP's automatic window size finder
finder = ClaSPSegmenter();
auto_window = finder.findDominantWindowSizes(ts, 'Offset', 0.1);

fprintf('  - Automatic window size: %d\n', auto_window);
fprintf('  - Dataset recommended size: %d\n', period_size);
fprintf('  - Using dataset recommendation: %d\n\n', period_size);

%% 6. ClaSP Segmentation
fprintf('Running ClaSP segmentation...\n');

% Create ClaSP segmenter with parameters matching Python
clasp = ClaSPSegmenter('PeriodLength', period_size, ...
                       'NumChangePoints', 4, ...
                       'ExclusionRadius', 0.05);

% Find change points
tic;
detected_cps = clasp.fitPredict(ts);
elapsed_time = toc;

fprintf('Segmentation complete in %.2f seconds.\n\n', elapsed_time);

% Sort change points for comparison
detected_cps = sort(detected_cps);

%% 7. Results Comparison
fprintf('Results Comparison:\n');
fprintf('==================\n');
fprintf('True change points:     [%s]\n', num2str(true_cps'));
fprintf('Detected change points: [%s]\n', num2str(detected_cps'));

% Calculate errors for each change point
errors = zeros(length(true_cps), 1);
for i = 1:min(length(true_cps), length(detected_cps))
    errors(i) = abs(true_cps(i) - detected_cps(i));
    fprintf('  CP%d error: %d samples (%.1f%%)\n', ...
            i, errors(i), 100*errors(i)/n);
end

% Calculate overall accuracy (within 5% tolerance)
tolerance = 0.05 * n;  % 5% of series length
accurate = sum(errors < tolerance);
accuracy = 100 * accurate / length(true_cps);
fprintf('\nAccuracy (5%% tolerance): %.1f%%\n', accuracy);

%% 8. Visualization of Segmentation Results
figure('Position', [100, 100, 1200, 600]);

% Top plot: Original segmentation
subplot(2,1,1);
plot(ts, 'b-', 'LineWidth', 0.8);
hold on;

% Add true change points
for i = 1:length(true_cps)
    xline(true_cps(i), 'g:', 'LineWidth', 2);
end

% Add detected change points
for i = 1:length(detected_cps)
    xline(detected_cps(i), 'r--', 'LineWidth', 2);
end

xlabel('Time');
ylabel('Value');
title('ClaSP Segmentation Results');
legend('Time Series', 'True CPs', 'Detected CPs', 'Location', 'best');
grid on;

% Bottom plot: Segmented regions with colors
subplot(2,1,2);
plot(ts, 'k-', 'LineWidth', 0.5, 'Color', [0.7, 0.7, 0.7]);
hold on;

% Color each segment
colors = lines(length(detected_cps) + 1);
segment_starts = [1; detected_cps];
segment_ends = [detected_cps - 1; n];

for i = 1:length(segment_starts)
    idx = segment_starts(i):segment_ends(i);
    plot(idx, ts(idx), 'LineWidth', 2, 'Color', colors(i,:));
end

% Add vertical lines for detected change points
for i = 1:length(detected_cps)
    xline(detected_cps(i), 'k--', 'LineWidth', 1.5);
end

xlabel('Time');
ylabel('Value');
title(sprintf('Segmented Time Series (%d segments)', length(detected_cps) + 1));
grid on;

%% 9. Segment Statistics
fprintf('\nSegment Analysis:\n');
fprintf('=================\n');

% Calculate statistics for each segment
for i = 1:length(segment_starts)
    idx = segment_starts(i):segment_ends(i);
    segment_mean = mean(ts(idx));
    segment_std = std(ts(idx));
    segment_length = length(idx);

    fprintf('Segment %d: [%d-%d]\n', i, segment_starts(i), segment_ends(i));
    fprintf('  Length: %d samples (%.1f%%)\n', segment_length, 100*segment_length/n);
    fprintf('  Mean: %.4f\n', segment_mean);
    fprintf('  Std Dev: %.4f\n\n', segment_std);
end

%% 10. Parameter Sensitivity Analysis
fprintf('Testing parameter sensitivity...\n');

% Test different exclusion radius values
exclusion_radii = [0.02, 0.05, 0.1, 0.15];
figure('Position', [100, 100, 1200, 800]);

for i = 1:length(exclusion_radii)
    er = exclusion_radii(i);

    % Run ClaSP with different exclusion radius
    clasp_test = ClaSPSegmenter('PeriodLength', period_size, ...
                                'NumChangePoints', 4, ...
                                'ExclusionRadius', er);
    cps_test = clasp_test.fitPredict(ts);

    % Plot results
    subplot(2, 2, i);
    plot(ts, 'b-', 'LineWidth', 0.5);
    hold on;

    % Add detected change points
    for cp = cps_test'
        xline(cp, 'r--', 'LineWidth', 1.5);
    end

    % Add true change points for reference
    for cp = true_cps'
        xline(cp, 'g:', 'LineWidth', 1);
    end

    title(sprintf('Exclusion Radius = %.2f', er));
    xlabel('Time');
    ylabel('Value');
    grid on;

    if i == 1
        legend('Time Series', 'Detected', 'True', 'Location', 'best');
    end
end

sgtitle('Effect of Exclusion Radius on Change Point Detection');

%% 11. ClaSP Profile Hierarchical Analysis
fprintf('\nHierarchical segmentation process:\n');
fprintf('===================================\n');

% Show how ClaSP hierarchically finds change points
figure('Position', [100, 100, 1200, 800]);

% Run ClaSP incrementally to show the process
for num_cps = 1:4
    clasp_hier = ClaSPSegmenter('PeriodLength', period_size, ...
                                'NumChangePoints', num_cps, ...
                                'ExclusionRadius', 0.05);
    cps_hier = clasp_hier.fitPredict(ts);

    subplot(2, 2, num_cps);
    plot(ts, 'b-', 'LineWidth', 0.8);
    hold on;

    % Add detected change points
    for j = 1:length(cps_hier)
        xline(cps_hier(j), 'r--', 'LineWidth', 2, ...
              'Label', sprintf('CP%d', j));
    end

    % Add true change points for reference
    for cp = true_cps'
        xline(cp, 'g:', 'LineWidth', 0.5);
    end

    title(sprintf('Finding %d Change Point(s)', num_cps));
    xlabel('Time');
    ylabel('Value');
    grid on;

    fprintf('With %d CPs: [%s]\n', num_cps, num2str(sort(cps_hier)'));
end

sgtitle('Hierarchical Change Point Detection Process');

%% 12. Summary and Conclusions
fprintf('\n========================================\n');
fprintf('              SUMMARY\n');
fprintf('========================================\n');
fprintf('Dataset: Electric Devices (n=%d)\n', n);
fprintf('Method: ClaSP (Classification Score Profile)\n');
fprintf('Parameters:\n');
fprintf('  - Window size: %d\n', period_size);
fprintf('  - Number of CPs: %d\n', 4);
fprintf('  - Exclusion radius: %.2f\n', 0.05);
fprintf('\nResults:\n');
fprintf('  - Processing time: %.2f seconds\n', elapsed_time);
fprintf('  - Detected CPs: [%s]\n', num2str(detected_cps'));
fprintf('  - Mean error: %.1f samples\n', mean(errors(errors>0)));
fprintf('  - Accuracy: %.1f%%\n', accuracy);
fprintf('\nKey Insights:\n');
fprintf('  - ClaSP successfully identifies major transitions in energy consumption\n');
fprintf('  - The hierarchical approach finds the most prominent changes first\n');
fprintf('  - Window size selection is critical for optimal performance\n');
fprintf('  - Exclusion radius prevents spurious boundary detections\n');
fprintf('========================================\n');

%% Save Results
% Save the results for comparison with Python
results = struct();
results.true_changepoints = true_cps;
results.detected_changepoints = detected_cps;
results.errors = errors;
results.accuracy = accuracy;
results.period_size = period_size;
results.profile = profile;
results.elapsed_time = elapsed_time;

save('electric_devices_results.mat', 'results');
fprintf('\nResults saved to electric_devices_results.mat\n');