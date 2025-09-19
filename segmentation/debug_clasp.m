% Debug script to understand ClaSP segmentation issue
addpath(genpath('.'));

% Create test data
ts = [sin(0.1*(1:300)), 2*sin(0.3*(301:600)), 0.5*cos(0.05*(601:1000))]';
fprintf('Time series length: %d\n', length(ts));

% Create ClaSP segmenter
clasp = ClaSPSegmenter('PeriodLength', 30, 'NumChangePoints', 2);

% Compute global profile
globalProfile = clasp.computeClaspProfile(ts, 30);
fprintf('Global profile length: %d\n', length(globalProfile));

% Find best change point manually to see what happens
exclusionZone = round(clasp.ExclusionRadius * length(ts));
fprintf('Exclusion zone size: %d\n', exclusionZone);

% Replicate the findBestChangePoint logic manually
validProfile = globalProfile;
startIdx = min(exclusionZone, length(globalProfile));
endIdx = max(1, length(globalProfile) - exclusionZone + 1);

fprintf('Exclusion start: %d, end: %d\n', startIdx, endIdx);

if startIdx >= 1
    validProfile(1:startIdx) = -inf;
end
if endIdx <= length(globalProfile)
    validProfile(endIdx:end) = -inf;
end

[maxScore, maxIdx] = max(validProfile);
fprintf('Best change point: index %d, score %.4f\n', maxIdx, maxScore);

% Now test what happens in the queue
fprintf('\n--- Testing queue logic ---\n');
rangeStart = 1;
rangeEnd = length(ts);
changePointIdx = maxIdx + rangeStart - 1;
fprintf('rangeStart: %d, maxIdx: %d, calculated changePointIdx: %d\n', rangeStart, maxIdx, changePointIdx);

% Test the actual fitPredict method
fprintf('\n--- Testing fitPredict ---\n');
foundCps = clasp.fitPredict(ts);
fprintf('Found change points: %s\n', mat2str(foundCps'));

% Debug the queue initialization issue
fprintf('\n--- Debug queue initialization ---\n');
fprintf('max(globalProfile): %.4f\n', max(globalProfile));
fprintf('globalProfile(maxIdx): %.4f\n', globalProfile(maxIdx));
fprintf('Are they equal? %d\n', max(globalProfile) == globalProfile(maxIdx));