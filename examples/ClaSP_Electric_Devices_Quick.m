%% ClaSP Segmentation - Electric Devices (Quick Version)
% A streamlined example using the Electric Devices dataset

%% Setup
clear; clc; close all;
addpath('../segmentation');
addpath('../testing/data');

fprintf('Loading Electric Devices dataset...\n');

%% Load Data
ts = readmatrix('electric_devices_timeseries.csv');
true_cps = readmatrix('electric_devices_changepoints.csv')';
period_size = readmatrix('electric_devices_period.txt');

fprintf('Dataset: %d samples, %d true change points\n', length(ts), length(true_cps));
fprintf('True CPs: [%s]\n\n', num2str(true_cps'));

%% Visualize Original Data
figure('Position', [100, 100, 1200, 400]);
plot(ts, 'b-', 'LineWidth', 0.8);
hold on;
for cp = true_cps'
    xline(cp, 'g:', 'LineWidth', 1.5);
end
xlabel('Time');
ylabel('Energy Consumption');
title('Electric Devices Time Series with True Change Points (green)');
grid on;

%% Run ClaSP Segmentation
fprintf('Running ClaSP segmentation (this may take a minute)...\n');
fprintf('Parameters: Period=%d, NumCPs=%d, ExclusionRadius=0.05\n\n', ...
        period_size, length(true_cps));

% Create segmenter
clasp = ClaSPSegmenter('PeriodLength', period_size, ...
                       'NumChangePoints', length(true_cps), ...
                       'ExclusionRadius', 0.05);

% Find change points
tic;
detected_cps = clasp.fitPredict(ts);
elapsed = toc;

detected_cps = sort(detected_cps);

fprintf('Segmentation complete in %.1f seconds\n\n', elapsed);

%% Results
fprintf('RESULTS:\n');
fprintf('========\n');
fprintf('True CPs:     [%s]\n', num2str(true_cps'));
fprintf('Detected CPs: [%s]\n\n', num2str(detected_cps'));

% Calculate errors
for i = 1:min(length(true_cps), length(detected_cps))
    error = abs(true_cps(i) - detected_cps(i));
    fprintf('CP%d: True=%d, Detected=%d, Error=%d (%.1f%%)\n', ...
            i, true_cps(i), detected_cps(i), error, 100*error/length(ts));
end

% Overall accuracy
tolerance = 0.05 * length(ts);  % 5% tolerance
errors = abs(true_cps - detected_cps(1:length(true_cps)));
accurate = sum(errors < tolerance);
fprintf('\nAccuracy (5%% tolerance): %.1f%%\n', 100*accurate/length(true_cps));

%% Visualization
figure('Position', [100, 100, 1200, 600]);

% Top: Comparison
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

% Bottom: Segmented view
subplot(2,1,2);
colors = lines(length(detected_cps) + 1);
segments = [1; detected_cps; length(ts)];

for i = 1:(length(segments)-1)
    idx = segments(i):segments(i+1);
    plot(idx, ts(idx), 'LineWidth', 1.5, 'Color', colors(i,:));
    hold on;
end

for cp = detected_cps'
    xline(cp, 'k--', 'LineWidth', 1);
end
xlabel('Time');
ylabel('Value');
title(sprintf('Segmented Time Series (%d segments)', length(detected_cps)+1));
grid on;

fprintf('\nVisualization complete!\n');