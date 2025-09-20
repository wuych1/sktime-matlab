classdef ClaSPSegmenter < handle
    % CLASPSEGMENTER ClaSP (Classification Score Profile) segmentation algorithm
    %
    % ClaSP segments time series by finding optimal change points using
    % k-nearest neighbor classification and ROC-AUC scoring. The algorithm
    % recursively identifies split points that best separate subsequences
    % into distinct classes.
    %
    % Key Features:
    %   - Automatic boundary exclusion to prevent spurious edge detections
    %   - Adjustable exclusion radius for controlling sensitivity
    %   - Priority queue-based recursive segmentation
    %   - Compatible with Python sktime implementation

    properties (Access = public)
        % PeriodLength - Window size for sliding window analysis (default: 10)
        %   Typically set to the expected period/pattern length in your data
        PeriodLength (1,1) double {mustBePositive, mustBeInteger} = 10

        % NumChangePoints - Number of change points to detect (default: 1)
        NumChangePoints (1,1) double {mustBeNonnegative, mustBeInteger} = 1

        % ExclusionRadius - Fraction of series length for exclusion zones (default: 0.05)
        %   Controls how close change points can be to each other and boundaries
        %   - 0.05-0.1: Standard setting, excludes 5-10% around each point
        %   - 0.02 or less: Use when boundary changes are expected
        ExclusionRadius (1,1) double {mustBeNonnegative} = 0.05
    end

    properties (SetAccess = private, GetAccess = public)
        Profiles
        Scores
    end

    properties (Access = private)
        transformer
        lastTimeSeries
        changePoints
    end

    methods
        function obj = ClaSPSegmenter(options)
            % Constructor
            arguments
                options.PeriodLength (1,1) double {mustBePositive, mustBeInteger} = 10
                options.NumChangePoints (1,1) double {mustBeNonnegative, mustBeInteger} = 1
                options.ExclusionRadius (1,1) double {mustBeNonnegative} = 0.05
            end

            obj.PeriodLength = options.PeriodLength;
            obj.NumChangePoints = options.NumChangePoints;
            obj.ExclusionRadius = options.ExclusionRadius;

            obj.transformer = ClaSPTransformer();
        end

        function changePoints = fitPredict(obj, timeSeries)
            % FITPREDICT Fit ClaSP model and predict change points
            %
            % Inputs:
            %   timeSeries - Numeric vector of time series data
            %
            % Outputs:
            %   changePoints - Column vector of detected change point indices
            %
            % Note: Change points near series boundaries (within ExclusionRadius)
            %       will be automatically filtered to avoid boundary artifacts

            validateattributes(timeSeries, {'numeric'}, {'vector', 'finite'});
            timeSeries = timeSeries(:);

            if length(timeSeries) < 2 * obj.PeriodLength
                error('ClaSPSegmenter:InsufficientData', ...
                    'Time series length must be at least 2 * PeriodLength (%d)', 2 * obj.PeriodLength);
            end

            obj.lastTimeSeries = timeSeries;

            [changePoints, scores, profiles] = obj.segmentation(timeSeries);

            obj.changePoints = changePoints;
            obj.Scores = scores;
            obj.Profiles = profiles;
        end

        function windowSizes = findDominantWindowSizes(obj, timeSeries, options)
            % FINDDOMINANTWINDOWSIZES Find optimal window sizes using FFT
            arguments
                obj
                timeSeries (:,1) double {mustBeFinite}
                options.Offset (1,1) double {mustBePositive} = 0.1
            end

            offset = options.Offset;
            timeSeries = timeSeries(:);
            n = length(timeSeries);

            Y = fft(timeSeries - mean(timeSeries));
            P = abs(Y).^2;

            [~, freqIdx] = maxk(P(2:floor(n/2)), 5);
            freqIdx = freqIdx + 1;

            periods = n ./ freqIdx;
            halfPeriods = periods / 2;

            minWindow = 20;
            maxWindow = floor(n * offset);
            validPeriods = halfPeriods(halfPeriods >= minWindow & halfPeriods <= maxWindow);

            if isempty(validPeriods)
                windowSizes = min(max(minWindow, 10), maxWindow);
            else
                windowSizes = round(validPeriods(1));
            end
        end
    end

    methods (Access = private)
        function [changePoints, scores, profiles] = segmentation(obj, timeSeries)
            % SEGMENTATION Exact implementation of Python _segmentation method

            fprintf('Starting segmentation...\n');

            queue = {};

            fprintf('Computing global profile...\n');
            [globalProfile, ~] = obj.transformer.transform(timeSeries, obj.PeriodLength, 'ExclusionRadius', obj.ExclusionRadius);

            [maxScore, maxIdx] = max(globalProfile);
            if isempty(maxIdx) || maxScore <= 0
                changePoints = [];
                scores = [];
                profiles = {};
                return;
            end

            profileRange = 1:length(timeSeries);
            queue{end+1} = struct('priority', -maxScore, 'range', profileRange, 'changePoint', maxIdx, 'profile', globalProfile);

            changePoints = [];
            scores = [];
            profiles = {};

            for idx = 1:obj.NumChangePoints
                fprintf('Finding change point %d/%d...\n', idx, obj.NumChangePoints);

                if isempty(queue)
                    fprintf('Queue empty, stopping early\n');
                    break;
                end

                priorities = cellfun(@(x) x.priority, queue);
                [~, bestIdx] = min(priorities);
                bestItem = queue{bestIdx};
                queue(bestIdx) = [];

                currentRange = bestItem.range;
                changePoint = bestItem.changePoint;
                fullProfile = bestItem.profile;

                globalChangePoint = currentRange(changePoint);

                if obj.isTrivialMatch(globalChangePoint, changePoints, timeSeries)
                    fprintf('Trivial match detected, skipping\n');
                    continue;
                end

                changePoints(end+1) = globalChangePoint;
                scores(end+1) = -bestItem.priority;
                profiles{end+1} = fullProfile;

                fprintf('Found change point at index %d with score %.4f\n', globalChangePoint, -bestItem.priority);

                if idx == obj.NumChangePoints
                    break;
                end

                if changePoint > 1
                    leftRange = currentRange(1):currentRange(changePoint-1);
                else
                    leftRange = [];
                end

                if changePoint < length(currentRange)
                    rightRange = currentRange(changePoint+1):currentRange(end);
                else
                    rightRange = [];
                end

                for ranges = {leftRange, rightRange}
                    subRange = ranges{1};

                    if isempty(subRange)
                        continue;
                    end

                    % Important: Don't use segment-based exclusion for feasibility check
                    % Just check if segment is large enough for window
                    if length(subRange) <= obj.PeriodLength
                        continue;
                    end

                    subSeries = timeSeries(subRange);

                    try
                        [localProfile, ~] = obj.transformer.transform(subSeries, obj.PeriodLength, 'ExclusionRadius', obj.ExclusionRadius);

                        % Don't apply additional exclusion here - profile already has it
                        validProfile = localProfile;

                        [localMaxScore, localMaxIdx] = max(localProfile);

                        % Don't require a minimum score - let the queue sorting handle priority
                        if ~isempty(localMaxIdx) && localMaxIdx > 0
                            queue{end+1} = struct('priority', -localMaxScore, 'range', subRange, 'changePoint', localMaxIdx, 'profile', localProfile);
                            fprintf('Added sub-range [%d:%d] to queue with score %.4f\n', subRange(1), subRange(end), localMaxScore);
                        end

                    catch ME
                        fprintf('Failed to compute profile for sub-range [%d:%d]: %s\n', subRange(1), subRange(end), ME.message);
                        continue;
                    end
                end
            end

            fprintf('Segmentation complete. Found %d change points.\n', length(changePoints));
        end

        function isTrivial = isTrivialMatch(obj, newChangePoint, existingChangePoints, timeSeries)
            % ISTRIVIALMATCH Check if change point is too close to existing ones
            % Matches Python implementation: includes boundaries in exclusion check

            % Add boundaries (start and end) to the change points list
            % This is critical for matching Python behavior!
            n = length(timeSeries);
            allChangePoints = [1; existingChangePoints(:); n];

            exclusionZone = round(obj.ExclusionRadius * n);

            % Check if candidate is within exclusion zone of any change point
            for cp = allChangePoints'
                leftBegin = max(1, cp - exclusionZone);
                rightEnd = min(n, cp + exclusionZone);
                if newChangePoint >= leftBegin && newChangePoint <= rightEnd
                    isTrivial = true;
                    return;
                end
            end

            isTrivial = false;
        end
    end
end