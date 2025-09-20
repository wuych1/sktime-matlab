%% ClaSP Quick Start Guide
% A simple introduction to time series segmentation with ClaSP

%% Setup
clear; clc; close all;
addpath('../segmentation');

%% Basic Example: Detecting a Change Point
% Create a simple time series with one change point

% Generate data: flat line that jumps to a different level
n = 200;
change_point = 100;
data = [ones(1, change_point)*2, ones(1, n-change_point)*5];
data = data + randn(1, n) * 0.2;  % Add some noise

% Create ClaSP segmenter
segmenter = ClaSPSegmenter('PeriodLength', 20, 'NumChangePoints', 1);

% Find change points
detected_cp = segmenter.fitPredict(data');

% Visualize
figure;
plot(data, 'LineWidth', 1.5);
hold on;
xline(detected_cp, 'r--', 'LineWidth', 2, 'Label', 'Detected');
xline(change_point, 'g:', 'LineWidth', 2, 'Label', 'True');
xlabel('Time');
ylabel('Value');
title('ClaSP Change Point Detection');
legend('show');
grid on;

fprintf('True change point: %d\n', change_point);
fprintf('Detected change point: %d\n', detected_cp);
fprintf('Detection error: %d samples\n\n', abs(detected_cp - change_point));

%% Understanding the Score Profile
% The ClaSP algorithm creates a "score profile" that shows how good
% each point would be as a change point

% Get the score profile from the segmenter
profile = segmenter.Profiles{1};

figure;
subplot(2,1,1);
plot(data);
xline(detected_cp, 'r--', 'LineWidth', 2);
title('Time Series');
ylabel('Value');
grid on;

subplot(2,1,2);
plot(profile);
hold on;
plot(detected_cp, profile(detected_cp), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('ClaSP Score Profile (higher = better change point)');
xlabel('Time');
ylabel('Classification Score');
grid on;

%% Multiple Change Points
% ClaSP can find multiple change points hierarchically

% Create data with 3 levels
data_multi = [ones(1, 100)*1, ones(1, 100)*4, ones(1, 100)*2];
data_multi = data_multi + randn(1, 300) * 0.15;

% Find 2 change points
segmenter_multi = ClaSPSegmenter('PeriodLength', 20, 'NumChangePoints', 2);
cps = segmenter_multi.fitPredict(data_multi');

figure;
plot(data_multi, 'LineWidth', 1.5);
hold on;
for cp = cps'
    xline(cp, 'r--', 'LineWidth', 2);
end
xlabel('Time');
ylabel('Value');
title('Multiple Change Points Detection');
legend('Data', 'Change Points');
grid on;

fprintf('Detected change points: %s\n', num2str(cps'));

%% Automatic Window Size Selection
% Use findDominantWindowSizes to automatically choose window size

% Create periodic signal with frequency change
t = linspace(0, 20, 400);
data_periodic = [sin(2*pi*2*t(1:200)), sin(2*pi*5*t(201:400))];

% Find optimal window size
finder = ClaSPSegmenter();
optimal_window = finder.findDominantWindowSizes(data_periodic');
fprintf('\nOptimal window size: %d\n', optimal_window);

% Use the optimal window size
segmenter_auto = ClaSPSegmenter('PeriodLength', optimal_window, 'NumChangePoints', 1);
cp_auto = segmenter_auto.fitPredict(data_periodic');

figure;
plot(data_periodic);
xline(cp_auto, 'r--', 'LineWidth', 2, 'Label', 'Detected');
xline(200, 'g:', 'LineWidth', 2, 'Label', 'True');
title(sprintf('Automatic Parameter Selection (Window Size = %d)', optimal_window));
xlabel('Time');
ylabel('Value');
legend('show');
grid on;

fprintf('Frequency change detected at: %d (true: 200)\n', cp_auto);