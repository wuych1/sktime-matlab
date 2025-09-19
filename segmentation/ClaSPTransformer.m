classdef ClaSPTransformer < handle
    % CLASPTRANSFORMER Core ClaSP transformation for profile computation
    %
    % ClaSPTransformer implements the core Classification Score Profile
    % transformation that computes segmentation scores across a time series
    % using k-nearest neighbors and binary classification metrics.
    %
    % USAGE:
    %   transformer = ClaSPTransformer();
    %   profile = transformer.transform(timeSeries, windowSize);
    %
    % METHODS:
    %   transform            - Compute ClaSP profile for time series
    %   slidingDotProduct    - Efficient sliding dot product computation
    %   computeKnnDistances  - Find k-nearest neighbors for subsequences
    %   calcKnnLabels        - Generate binary labels for split points
    %   calcProfile          - Compute classification score profile
    %
    % See also: ClaSPSegmenter

    properties (Access = private, Constant)
        DEFAULT_K = 3  % Default number of nearest neighbors
        INTERPOLATION_FACTOR = 4  % Profile interpolation factor
    end

    methods
        function obj = ClaSPTransformer()
            % CLASPTRANSFORMER Constructor for ClaSP transformer
        end

        function profile = transform(obj, timeSeries, windowSize, varargin)
            % TRANSFORM Compute ClaSP classification score profile
            %
            % SYNTAX:
            %   profile = obj.transform(timeSeries, windowSize)
            %   profile = obj.transform(timeSeries, windowSize, 'K', k)
            %
            % INPUT ARGUMENTS:
            %   timeSeries - Input time series (numeric vector)
            %   windowSize - Sliding window size (positive integer)
            %   'K'        - Number of nearest neighbors (default: 3)
            %
            % OUTPUT ARGUMENTS:
            %   profile - Classification score profile (numeric vector)

            % Parse input arguments
            p = inputParser;
            addRequired(p, 'timeSeries', @(x) validateattributes(x, {'numeric'}, {'vector'}));
            addRequired(p, 'windowSize', @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            addParameter(p, 'K', obj.DEFAULT_K, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            parse(p, timeSeries, windowSize, varargin{:});

            k = p.Results.K;
            timeSeries = timeSeries(:);  % Ensure column vector
            n = length(timeSeries);

            if windowSize >= n - 1
                error('ClaSPTransformer:InvalidWindowSize', ...
                    'Window size (%d) must be smaller than the time series length (%d)', windowSize, n-1);
            end

            % Main ClaSP algorithm
            profile = obj.clasp(timeSeries, windowSize, k);
        end

        function profile = clasp(obj, timeSeries, windowSize, k)
            % CLASP Core ClaSP algorithm implementation

            n = length(timeSeries);
            numSubseq = n - windowSize + 1;

            % Step 1: Extract subsequences and compute statistics
            [subsequences, means, stds] = obj.extractSubsequences(timeSeries, windowSize);

            % Step 2: Compute distances using normalized subsequences
            distances = obj.computeNormalizedDistances(subsequences, windowSize);

            % Step 3: Find k-nearest neighbors with exclusion zones
            knnIndices = obj.findKNNWithExclusion(distances, k, windowSize);

            % Step 4: Compute classification labels for each split point
            labels = obj.computeClassificationLabels(knnIndices, numSubseq);

            % Step 5: Calculate profile using simple scoring
            profile = obj.calculateProfile(labels, k);

            % Step 6: Interpolate to handle NaN values
            profile = obj.interpolateNaNValues(profile);
        end

        function [subsequences, means, stds] = extractSubsequences(obj, timeSeries, windowSize)
            % EXTRACTSUBSEQUENCES Extract and normalize subsequences

            n = length(timeSeries);
            numSubseq = n - windowSize + 1;
            subsequences = zeros(numSubseq, windowSize);
            means = zeros(numSubseq, 1);
            stds = zeros(numSubseq, 1);

            for i = 1:numSubseq
                subseq = timeSeries(i:i+windowSize-1);
                means(i) = mean(subseq);
                stds(i) = std(subseq);

                if stds(i) == 0
                    subsequences(i, :) = zeros(1, windowSize);
                else
                    subsequences(i, :) = (subseq - means(i)) / stds(i);
                end
            end
        end


        function distances = computeNormalizedDistances(obj, subsequences, windowSize)
            % COMPUTENORMALIZEDDISTANCES Compute Euclidean distances between normalized subsequences

            numSubseq = size(subsequences, 1);
            distances = zeros(numSubseq, numSubseq);

            % Compute pairwise Euclidean distances
            for i = 1:numSubseq
                for j = i+1:numSubseq
                    dist = norm(subsequences(i, :) - subsequences(j, :));
                    distances(i, j) = dist;
                    distances(j, i) = dist;
                end
            end
        end

        function knnIndices = findKNNWithExclusion(obj, distances, k, windowSize)
            % FINDKNNWITHEXCLUSION Find k-nearest neighbors with exclusion zones

            numSubseq = size(distances, 1);
            knnIndices = zeros(numSubseq, k);

            % Exclusion zone (slack parameter from Python)
            exclusionZone = max(1, round(0.25 * windowSize));

            for i = 1:numSubseq
                % Create working copy of distances
                workingDists = distances(i, :);

                % Exclude self-match
                workingDists(i) = inf;

                % Apply exclusion zone
                startExcl = max(1, i - exclusionZone);
                endExcl = min(numSubseq, i + exclusionZone);
                workingDists(startExcl:endExcl) = inf;

                % Find k smallest distances using partial sort
                [~, sortedIdx] = sort(workingDists);
                validIdx = sortedIdx(~isinf(workingDists(sortedIdx)));

                if length(validIdx) >= k
                    knnIndices(i, :) = validIdx(1:k);
                else
                    % Fallback: fill with available neighbors
                    numValid = length(validIdx);
                    if numValid > 0
                        knnIndices(i, 1:numValid) = validIdx;
                        knnIndices(i, numValid+1:k) = validIdx(end);  % Repeat last
                    else
                        knnIndices(i, :) = 1;  % Emergency fallback
                    end
                end
            end
        end

        function labels = computeClassificationLabels(obj, knnIndices, numSubseq)
            % COMPUTECLASSIFICATIONLABELS Generate binary labels for splits


            [numSubseq, k] = size(knnIndices);
            labels = zeros(numSubseq, numSubseq);

            % For each subsequence and each potential split point
            for i = 1:numSubseq
                for splitPoint = 1:numSubseq
                    leftCount = 0;
                    % Count how many k-nearest neighbors are to the left of split
                    for j = 1:k
                        if knnIndices(i, j) < splitPoint
                            leftCount = leftCount + 1;
                        end
                    end
                    labels(i, splitPoint) = leftCount / k;
                end
            end
        end

        function profile = calculateProfile(obj, labels, k)
            % CALCULATEPROFILE Compute ClaSP profile using simple classification metric

            [numSubseq, numSplits] = size(labels);
            profile = zeros(1, numSplits);

            for splitIdx = 1:numSplits
                % Get labels for this split point
                splitLabels = labels(:, splitIdx);

                % Calculate how well this split separates the data
                % Higher variance in labels indicates better separation
                if var(splitLabels) > 0
                    % Use the variance as a measure of separation quality
                    profile(splitIdx) = var(splitLabels) * 4;  % Scale up
                else
                    profile(splitIdx) = 0;
                end
            end
        end

        function profile = interpolateNaNValues(obj, profile)
            % INTERPOLATENANVALUES Handle NaN values in profile

            % Find NaN values
            nanIdx = isnan(profile);

            if ~any(nanIdx)
                return;  % No NaN values
            end

            % Simple linear interpolation for NaN values
            validIdx = find(~nanIdx);
            if length(validIdx) >= 2
                profile(nanIdx) = interp1(validIdx, profile(validIdx), find(nanIdx), 'linear', 'extrap');
            else
                profile(nanIdx) = 0.5;  % Default value
            end
        end
    end

    methods (Access = private)
        function distMatrix = computePairwiseDistances(obj, data)
            % COMPUTEPAIRWISEDISTANCES Fallback pairwise distance computation
            %
            % This method provides a fallback when pdist2 is not available

            n = size(data, 1);
            distMatrix = zeros(n, n);

            for i = 1:n
                for j = i+1:n
                    dist = norm(data(i, :) - data(j, :));
                    distMatrix(i, j) = dist;
                    distMatrix(j, i) = dist;
                end
            end
        end

        function auc = computeRocAuc(obj, scores, labels)
            % COMPUTEROCAUC Compute Area Under ROC Curve
            %
            % SYNTAX:
            %   auc = obj.computeRocAuc(scores, labels)
            %
            % INPUT ARGUMENTS:
            %   scores - Prediction scores (numeric vector)
            %   labels - True binary labels (logical vector)
            %
            % OUTPUT ARGUMENTS:
            %   auc - Area under ROC curve

            if length(unique(labels)) < 2
                auc = 0.5;  % Random performance for single class
                return;
            end

            % Sort scores in descending order
            [sortedScores, sortOrder] = sort(scores, 'descend');
            sortedLabels = labels(sortOrder);

            % Calculate ROC curve points
            numPos = sum(sortedLabels);
            numNeg = length(sortedLabels) - numPos;

            if numPos == 0 || numNeg == 0
                auc = 0.5;
                return;
            end

            tpr = cumsum(sortedLabels) / numPos;  % True positive rate
            fpr = cumsum(~sortedLabels) / numNeg; % False positive rate

            % Add (0,0) point
            tpr = [0; tpr];
            fpr = [0; fpr];

            % Calculate AUC using trapezoidal rule
            auc = trapz(fpr, tpr);

            % Ensure AUC is in [0, 1] range
            auc = max(0, min(1, auc));
        end
    end
end