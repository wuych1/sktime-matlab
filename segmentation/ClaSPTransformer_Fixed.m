classdef ClaSPTransformer_Fixed < handle
    % CLASPTRANSFORMER_FIXED Corrected ClaSP transformation implementation
    %
    % This is a corrected implementation following the actual ClaSP algorithm
    % from the Python reference implementation.

    properties (Access = private, Constant)
        DEFAULT_K = 3
    end

    methods
        function obj = ClaSPTransformer_Fixed()
            % Constructor
        end

        function profile = transform(obj, timeSeries, windowSize, varargin)
            % TRANSFORM Compute ClaSP classification score profile

            p = inputParser;
            addRequired(p, 'timeSeries', @(x) validateattributes(x, {'numeric'}, {'vector'}));
            addRequired(p, 'windowSize', @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            addParameter(p, 'K', obj.DEFAULT_K, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            parse(p, timeSeries, windowSize, varargin{:});

            k = p.Results.K;
            timeSeries = timeSeries(:);
            n = length(timeSeries);

            if windowSize >= n - 1
                error('ClaSPTransformer_Fixed:InvalidWindowSize', ...
                    'Window size (%d) must be smaller than the time series length (%d)', windowSize, n-1);
            end

            numSubseq = n - windowSize + 1;

            % Step 1: Extract and normalize subsequences
            subsequences = obj.extractSubsequences(timeSeries, windowSize);

            % Step 2: Compute distance matrix
            distMatrix = obj.computeDistanceMatrix(subsequences);

            % Step 3: Find k-nearest neighbors for each subsequence
            knnIndices = obj.findKNearestNeighbors(distMatrix, k, windowSize);

            % Step 4: Compute ClaSP profile using binary classification
            profile = obj.computeClaspProfile(knnIndices, numSubseq);
        end

        function subsequences = extractSubsequences(obj, timeSeries, windowSize)
            % Extract and z-normalize subsequences

            n = length(timeSeries);
            numSubseq = n - windowSize + 1;
            subsequences = zeros(numSubseq, windowSize);

            for i = 1:numSubseq
                subseq = timeSeries(i:i+windowSize-1);
                mu = mean(subseq);
                sigma = std(subseq);

                if sigma > 0
                    subsequences(i, :) = (subseq - mu) / sigma;
                else
                    subsequences(i, :) = zeros(1, windowSize);
                end
            end
        end

        function distMatrix = computeDistanceMatrix(obj, subsequences)
            % Compute pairwise Euclidean distances

            numSubseq = size(subsequences, 1);
            distMatrix = zeros(numSubseq, numSubseq);

            for i = 1:numSubseq
                for j = i+1:numSubseq
                    dist = norm(subsequences(i, :) - subsequences(j, :));
                    distMatrix(i, j) = dist;
                    distMatrix(j, i) = dist;
                end
            end
        end

        function knnIndices = findKNearestNeighbors(obj, distMatrix, k, windowSize)
            % Find k-nearest neighbors with exclusion zone

            numSubseq = size(distMatrix, 1);
            knnIndices = zeros(numSubseq, k);
            exclusionZone = round(windowSize / 4);  % 25% of window size

            for i = 1:numSubseq
                % Copy distances for this subsequence
                dists = distMatrix(i, :);

                % Exclude self and neighbors within exclusion zone
                dists(i) = inf;
                startExcl = max(1, i - exclusionZone);
                endExcl = min(numSubseq, i + exclusionZone);
                dists(startExcl:endExcl) = inf;

                % Find k smallest distances
                [~, sortedIdx] = sort(dists);
                validIdx = sortedIdx(~isinf(dists(sortedIdx)));

                if length(validIdx) >= k
                    knnIndices(i, :) = validIdx(1:k);
                else
                    % Not enough neighbors - fill with what we have
                    numValid = length(validIdx);
                    if numValid > 0
                        knnIndices(i, 1:numValid) = validIdx;
                        % Repeat last valid index for remaining slots
                        knnIndices(i, numValid+1:k) = validIdx(end);
                    else
                        % No valid neighbors - use distant ones
                        [~, fallbackIdx] = sort(distMatrix(i, :));
                        knnIndices(i, :) = fallbackIdx(1:k);
                    end
                end
            end
        end

        function profile = computeClaspProfile(obj, knnIndices, numSubseq)
            % Compute ClaSP profile using binary classification at each split point

            [numSubseq, k] = size(knnIndices);
            profile = zeros(1, numSubseq);

            for splitPoint = 1:numSubseq
                % For each subsequence, count how many of its k-NN are to the left of the split
                leftCounts = zeros(numSubseq, 1);

                for i = 1:numSubseq
                    leftCount = 0;
                    for j = 1:k
                        if knnIndices(i, j) < splitPoint
                            leftCount = leftCount + 1;
                        end
                    end
                    leftCounts(i) = leftCount / k;  % Normalize to [0, 1]
                end

                % Create binary classification problem
                % True labels: 1 if subsequence index < splitPoint, 0 otherwise
                trueLabels = (1:numSubseq)' < splitPoint;

                % Predicted scores: proportion of k-NN to the left
                predictedScores = leftCounts;

                % Compute ROC-AUC as the classification score
                if sum(trueLabels) > 0 && sum(trueLabels) < numSubseq
                    auc = obj.computeROCAUC(predictedScores, trueLabels);
                    % ClaSP uses AUC - 0.5 to center around 0, then takes absolute value
                    profile(splitPoint) = abs(auc - 0.5);
                else
                    profile(splitPoint) = 0;  % Cannot classify with single class
                end
            end
        end

        function auc = computeROCAUC(obj, scores, labels)
            % Compute Area Under ROC Curve

            % Handle edge cases
            if length(unique(labels)) < 2
                auc = 0.5;
                return;
            end

            % Sort by scores (descending)
            [~, sortIdx] = sort(scores, 'descend');
            sortedLabels = labels(sortIdx);

            % Calculate cumulative TP and FP
            numPos = sum(labels);
            numNeg = length(labels) - numPos;

            if numPos == 0 || numNeg == 0
                auc = 0.5;
                return;
            end

            cumulativeTP = cumsum(sortedLabels);
            cumulativeFP = cumsum(~sortedLabels);

            % Calculate TPR and FPR
            tpr = [0; cumulativeTP / numPos];
            fpr = [0; cumulativeFP / numNeg];

            % Calculate AUC using trapezoidal rule
            auc = trapz(fpr, tpr);

            % Ensure AUC is in [0, 1]
            auc = max(0, min(1, auc));
        end
    end
end