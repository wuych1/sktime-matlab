function changePoints = debug_segmentation(obj, timeSeries, startIdx, endIdx)
    % Debug version of segmentation method

    fprintf('=== Starting segmentation ===\n');
    fprintf('startIdx: %d, endIdx: %d\n', startIdx, endIdx);

    if obj.NumChangePoints <= 0
        changePoints = [];
        return;
    end

    % Initialize with global profile
    globalProfile = obj.computeClaspProfile(timeSeries, obj.PeriodLength);
    fprintf('Global profile computed, length: %d\n', length(globalProfile));

    % Priority queue: stores structs with fields
    globalMaxIdx = obj.findBestChangePoint(globalProfile, round(obj.ExclusionRadius * length(timeSeries)));
    fprintf('Global max index: %d\n', globalMaxIdx);

    if isempty(globalMaxIdx)
        changePoints = [];
        return;
    end

    queue = {struct('score', -max(globalProfile), 'start', startIdx, 'end', endIdx, 'profile', globalProfile, 'cpIdx', globalMaxIdx)};
    changePoints = [];

    fprintf('Queue initialized with cpIdx: %d\n', globalMaxIdx);

    for i = 1:obj.NumChangePoints
        fprintf('\n--- Iteration %d ---\n', i);

        if isempty(queue)
            fprintf('Queue is empty, breaking\n');
            break;
        end

        % Find highest scoring item (most negative score)
        scores = cellfun(@(x) x.score, queue);
        [~, bestIdx] = min(scores);  % Min because we use negative scores
        bestItem = queue{bestIdx};
        queue(bestIdx) = [];

        % Extract information
        rangeStart = bestItem.start;
        rangeEnd = bestItem.end;
        profile = bestItem.profile;
        changePointIdx = round(bestItem.cpIdx + rangeStart - 1);  % Adjust to global index

        fprintf('Best item - start: %d, end: %d, cpIdx: %d\n', rangeStart, rangeEnd, bestItem.cpIdx);
        fprintf('Calculated changePointIdx: %d\n', changePointIdx);

        % Check if this is a trivial match
        if obj.isTrivialMatch(changePointIdx, changePoints, timeSeries)
            fprintf('Trivial match, skipping\n');
            continue;
        end

        changePoints(end + 1) = changePointIdx;
        fprintf('Added change point: %d\n', changePointIdx);
        fprintf('Current change points: %s\n', mat2str(changePoints));

        % Split into left and right ranges and compute local profiles
        leftStart = rangeStart;
        leftEnd = changePointIdx - 1;
        rightStart = changePointIdx + 1;
        rightEnd = rangeEnd;

        fprintf('Left range: %d to %d, Right range: %d to %d\n', leftStart, leftEnd, rightStart, rightEnd);

        % Rest of the method...
        % (I'll skip the queue addition parts for now to focus on the main issue)
        break; % Just process first iteration for debugging
    end

    fprintf('Final change points: %s\n', mat2str(changePoints));
end