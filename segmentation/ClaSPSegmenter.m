classdef ClaSPSegmenter < handle
    % CLASPSEGMENTER ClaSP (Classification Score Profile) segmentation algorithm

    properties (Access = public)
        PeriodLength (1,1) double {mustBePositive, mustBeInteger} = 10
        NumChangePoints (1,1) double {mustBeNonnegative, mustBeInteger} = 1
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
        function obj = ClaSPSegmenter(varargin)
            % Constructor
            p = inputParser;
            addParameter(p, 'PeriodLength', 10, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            addParameter(p, 'NumChangePoints', 1, @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative', 'integer'}));
            addParameter(p, 'ExclusionRadius', 0.05, @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative'}));
            parse(p, varargin{:});

            obj.PeriodLength = p.Results.PeriodLength;
            obj.NumChangePoints = p.Results.NumChangePoints;
            obj.ExclusionRadius = p.Results.ExclusionRadius;

            obj.transformer = ClaSPTransformer();
        end

        function changePoints = fitPredict(obj, timeSeries)
            % FITPREDICT Fit ClaSP model and predict change points

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

        function windowSizes = findDominantWindowSizes(obj, timeSeries, varargin)
            % FINDDOMINANTWINDOWSIZES Find optimal window sizes using FFT

            p = inputParser;
            addRequired(p, 'timeSeries', @(x) validateattributes(x, {'numeric'}, {'vector'}));
            addParameter(p, 'Offset', 0.1, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
            parse(p, timeSeries, varargin{:});

            offset = p.Results.Offset;
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

                    exclusionZone = round(length(subRange) * obj.ExclusionRadius);
                    if length(subRange) - obj.PeriodLength <= 2
                        continue;
                    end

                    subSeries = timeSeries(subRange);

                    try
                        [localProfile, ~] = obj.transformer.transform(subSeries, obj.PeriodLength, 'ExclusionRadius', obj.ExclusionRadius);

                        validProfile = localProfile;
                        if exclusionZone > 0
                            validProfile(1:min(exclusionZone, length(validProfile))) = -inf;
                            endIdx = max(1, length(validProfile) - exclusionZone + 1);
                            if endIdx <= length(validProfile)
                                validProfile(endIdx:end) = -inf;
                            end
                        end

                        [localMaxScore, localMaxIdx] = max(validProfile);

                        if ~isempty(localMaxIdx) && localMaxScore > -inf && localMaxIdx > 0
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

            if isempty(existingChangePoints)
                isTrivial = false;
                return;
            end

            exclusionZone = round(obj.ExclusionRadius * length(timeSeries));
            distances = abs(existingChangePoints - newChangePoint);
            isTrivial = any(distances < exclusionZone);
        end
    end
end