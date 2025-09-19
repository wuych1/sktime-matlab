classdef ClaSPSegmenter < handle
    % CLASPSEGMENTER Classification Score Profile Time Series Segmentation
    %
    % ClaSP (Classification Score Profile) is a domain-agnostic time series
    % segmentation algorithm that detects change points by computing a profile
    % of classification scores across the time series.
    %
    % USAGE:
    %   clasp = ClaSPSegmenter('PeriodLength', 10, 'NumChangePoints', 5);
    %   foundCps = clasp.fitPredict(timeSeries);
    %   profiles = clasp.Profiles;
    %   scores = clasp.Scores;
    %
    % PROPERTIES:
    %   PeriodLength     - Window size for sliding analysis (default: 10)
    %   NumChangePoints  - Number of change points to detect (default: 1)
    %   ExclusionRadius  - Minimum distance between change points (default: 0.05)
    %   Profiles         - Computed classification score profiles (read-only)
    %   Scores           - Change point scores (read-only)
    %
    % METHODS:
    %   fitPredict       - Fit model and predict change points
    %   findDominantWindowSizes - Determine optimal window sizes using FFT
    %   computeClaspProfile     - Compute classification score profile
    %
    % REFERENCES:
    %   [1] Arik Ermshaus, Patrick Schäfer, and Ulf Leser. ClaSP: parameter-free
    %       time series segmentation. Data Mining and Knowledge Discovery, 2023.
    %
    % See also: ClaSPTransformer

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
            % CLASPSEGMENTER Constructor for ClaSP segmentation object
            %
            % SYNTAX:
            %   obj = ClaSPSegmenter()
            %   obj = ClaSPSegmenter('Name', Value, ...)
            %
            % INPUT ARGUMENTS:
            %   'PeriodLength'     - Sliding window size (default: 10)
            %   'NumChangePoints'  - Number of change points to find (default: 1)
            %   'ExclusionRadius'  - Exclusion radius as fraction of series length (default: 0.05)
            %
            % OUTPUT ARGUMENTS:
            %   obj - ClaSPSegmenter object
            %
            % EXAMPLE:
            %   clasp = ClaSPSegmenter('PeriodLength', 20, 'NumChangePoints', 3);

            % Parse input arguments
            p = inputParser;
            addParameter(p, 'PeriodLength', 10, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            addParameter(p, 'NumChangePoints', 1, @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative', 'integer'}));
            addParameter(p, 'ExclusionRadius', 0.05, @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative'}));
            parse(p, varargin{:});

            obj.PeriodLength = p.Results.PeriodLength;
            obj.NumChangePoints = p.Results.NumChangePoints;
            obj.ExclusionRadius = p.Results.ExclusionRadius;

            % Initialize transformer
            obj.transformer = ClaSPTransformer_Fixed();
        end

        function changePoints = fitPredict(obj, timeSeries)
            % FITPREDICT Fit ClaSP model and predict change points
            %
            % SYNTAX:
            %   changePoints = obj.fitPredict(timeSeries)
            %
            % INPUT ARGUMENTS:
            %   timeSeries - Input time series data (numeric vector)
            %
            % OUTPUT ARGUMENTS:
            %   changePoints - Detected change point indices (vector)
            %
            % EXAMPLE:
            %   ts = randn(1000, 1);
            %   clasp = ClaSPSegmenter('NumChangePoints', 3);
            %   cps = clasp.fitPredict(ts);

            % Validate input
            validateattributes(timeSeries, {'numeric'}, {'vector', 'finite'});
            timeSeries = timeSeries(:);  % Ensure column vector

            if length(timeSeries) < 2 * obj.PeriodLength
                error('ClaSPSegmenter:InsufficientData', ...
                    'Time series length must be at least 2 * PeriodLength (%d)', 2 * obj.PeriodLength);
            end

            obj.lastTimeSeries = timeSeries;

            % Find optimal window sizes if not specified
            if obj.PeriodLength <= 0
                obj.PeriodLength = obj.findDominantWindowSizes(timeSeries);
            end

            % Perform segmentation
            changePoints = obj.segmentation(timeSeries, 1, length(timeSeries));
            obj.changePoints = sort(changePoints);

            % Store results
            obj.Scores = obj.computeChangePointScores(timeSeries, changePoints);
        end

        function windowSizes = findDominantWindowSizes(obj, timeSeries, varargin)
            % FINDDOMINANTWINDOWSIZES Find optimal window sizes using FFT analysis
            %
            % SYNTAX:
            %   windowSizes = obj.findDominantWindowSizes(timeSeries)
            %   windowSizes = obj.findDominantWindowSizes(timeSeries, 'Offset', offset)
            %
            % INPUT ARGUMENTS:
            %   timeSeries - Input time series (numeric vector)
            %   'Offset'   - Fraction of series length for max window (default: 0.1)
            %
            % OUTPUT ARGUMENTS:
            %   windowSizes - Optimal window size(s)

            p = inputParser;
            addRequired(p, 'timeSeries', @(x) validateattributes(x, {'numeric'}, {'vector'}));
            addParameter(p, 'Offset', 0.1, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
            parse(p, timeSeries, varargin{:});

            offset = p.Results.Offset;
            timeSeries = timeSeries(:);
            n = length(timeSeries);

            % Compute FFT
            Y = fft(timeSeries - mean(timeSeries));
            P = abs(Y).^2;

            % Find dominant frequencies (excluding DC component)
            [~, freqIdx] = maxk(P(2:floor(n/2)), 5);
            freqIdx = freqIdx + 1;  % Adjust for skipped DC component

            % Convert to periods (window sizes) - Python returns half the period
            periods = n ./ freqIdx;
            halfPeriods = periods / 2;

            % Filter periods within reasonable range
            minWindow = 20;
            maxWindow = floor(n * offset);
            validPeriods = halfPeriods(halfPeriods >= minWindow & halfPeriods <= maxWindow);

            if isempty(validPeriods)
                windowSizes = min(max(minWindow, 10), maxWindow);
            else
                windowSizes = round(validPeriods(1));
            end
        end

        function profile = computeClaspProfile(obj, timeSeries, windowSize)
            % COMPUTECLASPPROFILE Compute ClaSP score profile for given window size
            %
            % SYNTAX:
            %   profile = obj.computeClaspProfile(timeSeries, windowSize)
            %
            % INPUT ARGUMENTS:
            %   timeSeries - Input time series (numeric vector)
            %   windowSize - Sliding window size (positive integer)
            %
            % OUTPUT ARGUMENTS:
            %   profile - Classification score profile (numeric vector)

            validateattributes(timeSeries, {'numeric'}, {'vector'});
            validateattributes(windowSize, {'numeric'}, {'scalar', 'positive', 'integer'});

            timeSeries = timeSeries(:);
            profile = obj.transformer.transform(timeSeries, windowSize);

            % Store profile for later access
            if isempty(obj.Profiles)
                obj.Profiles = profile;
            else
                obj.Profiles = [obj.Profiles, profile];
            end
        end
    end

    methods (Access = private)
        function changePoints = segmentation(obj, timeSeries, startIdx, endIdx)
            % SEGMENTATION Recursive segmentation algorithm following Python implementation

            if obj.NumChangePoints <= 0
                changePoints = [];
                return;
            end

            % Initialize with global profile
            globalProfile = obj.computeClaspProfile(timeSeries, obj.PeriodLength);

            % Priority queue: stores structs with fields
            globalMaxIdx = obj.findBestChangePoint(globalProfile, round(obj.ExclusionRadius * length(timeSeries)));

            if isempty(globalMaxIdx)
                changePoints = [];
                return;
            end

            queue = {struct('score', -max(globalProfile), 'start', startIdx, 'end', endIdx, 'profile', globalProfile, 'cpIdx', globalMaxIdx)};
            changePoints = [];

            for i = 1:obj.NumChangePoints
                if isempty(queue)
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

                % Check if this is a trivial match
                if obj.isTrivialMatch(changePointIdx, changePoints, timeSeries)
                    continue;
                end

                changePoints(end + 1) = changePointIdx;

                % Split into left and right ranges and compute local profiles
                leftStart = rangeStart;
                leftEnd = changePointIdx - 1;
                rightStart = changePointIdx + 1;
                rightEnd = rangeEnd;

                % Add left range if large enough
                leftLen = leftEnd - leftStart + 1;
                if leftLen >= 3 * obj.PeriodLength  % More conservative check
                    leftStart = round(leftStart);
                    leftEnd = round(leftEnd);
                    leftSeries = timeSeries(leftStart:leftEnd);

                    % Only compute profile if we have enough data
                    if length(leftSeries) > obj.PeriodLength
                        leftProfile = obj.computeClaspProfile(leftSeries, obj.PeriodLength);
                        leftMaxIdx = obj.findBestChangePoint(leftProfile, round(obj.ExclusionRadius * length(leftSeries)));

                        if ~isempty(leftMaxIdx) && leftMaxIdx > 0 && leftMaxIdx <= length(leftProfile)
                            leftMaxIdx = round(leftMaxIdx);
                            if leftMaxIdx >= 1 && leftMaxIdx <= length(leftProfile)
                                leftMaxScore = leftProfile(leftMaxIdx);
                                queue{end + 1} = struct('score', -leftMaxScore, 'start', leftStart, 'end', leftEnd, 'profile', leftProfile, 'cpIdx', leftMaxIdx);
                            end
                        end
                    end
                end

                % Add right range if large enough
                rightLen = rightEnd - rightStart + 1;
                if rightLen >= 3 * obj.PeriodLength  % More conservative check
                    rightStart = round(rightStart);
                    rightEnd = round(rightEnd);
                    rightSeries = timeSeries(rightStart:rightEnd);

                    % Only compute profile if we have enough data
                    if length(rightSeries) > obj.PeriodLength
                        rightProfile = obj.computeClaspProfile(rightSeries, obj.PeriodLength);
                        rightMaxIdx = obj.findBestChangePoint(rightProfile, round(obj.ExclusionRadius * length(rightSeries)));

                        if ~isempty(rightMaxIdx) && rightMaxIdx > 0 && rightMaxIdx <= length(rightProfile)
                            rightMaxIdx = round(rightMaxIdx);
                            if rightMaxIdx >= 1 && rightMaxIdx <= length(rightProfile)
                                rightMaxScore = rightProfile(rightMaxIdx);
                                queue{end + 1} = struct('score', -rightMaxScore, 'start', rightStart, 'end', rightEnd, 'profile', rightProfile, 'cpIdx', rightMaxIdx);
                            end
                        end
                    end
                end
            end
        end

        function [maxScore, maxIdx] = findBestChangePoint(obj, profile, exclusionZone)
            % FINDBESTCHANGEPOINT Find highest scoring change point in profile

            exclusionZone = max(1, round(exclusionZone));

            if length(profile) < 2 * exclusionZone
                maxScore = [];
                maxIdx = [];
                return;
            end

            % Apply exclusion zone at boundaries
            validProfile = profile;
            startIdx = min(exclusionZone, length(profile));
            endIdx = max(1, length(profile) - exclusionZone + 1);

            if startIdx >= 1
                validProfile(1:startIdx) = -inf;
            end
            if endIdx <= length(profile)
                validProfile(endIdx:end) = -inf;
            end

            [maxScore, maxIdx] = max(validProfile);

            if isempty(maxScore) || maxScore == -inf || isnan(maxScore)
                maxScore = [];
                maxIdx = [];
            elseif ~isfinite(maxIdx) || maxIdx < 1 || maxIdx > length(profile)
                maxScore = [];
                maxIdx = [];
            else
                maxIdx = round(maxIdx);  % Ensure integer
            end
        end

        function isTrivial = isTrivialMatch(obj, newChangePoint, existingChangePoints, timeSeries)
            % ISTRIVIALMATCH Check if change point is too close to existing ones

            if isempty(existingChangePoints)
                isTrivial = false;
                return;
            end

            exclusionRadius = round(obj.ExclusionRadius * length(timeSeries));
            distances = abs(existingChangePoints - newChangePoint);
            isTrivial = any(distances < exclusionRadius);
        end

        function scores = computeChangePointScores(obj, timeSeries, changePoints)
            % COMPUTECHANGEPOINTSCORES Compute scores for detected change points

            scores = zeros(size(changePoints));

            for i = 1:length(changePoints)
                cp = changePoints(i);
                windowStart = max(1, cp - obj.PeriodLength);
                windowEnd = min(length(timeSeries), cp + obj.PeriodLength);

                subSeriesLen = windowEnd - windowStart + 1;
                if subSeriesLen > obj.PeriodLength  % Need more than window size
                    subSeries = timeSeries(windowStart:windowEnd);
                    try
                        profile = obj.computeClaspProfile(subSeries, obj.PeriodLength);
                        localIdx = cp - windowStart + 1;

                        if localIdx > 0 && localIdx <= length(profile)
                            scores(i) = profile(localIdx);
                        end
                    catch
                        scores(i) = 0;  % Default score if computation fails
                    end
                else
                    scores(i) = 0;  % Default score for too-small segments
                end
            end
        end
    end
end