function [knnIndices, distances] = computeKnnDistances(timeSeries, windowSize, k, varargin)
    % COMPUTEKNNDISTANCES Find k-nearest neighbors for time series subsequences
    %
    % This function finds the k-nearest neighbors for each subsequence in a
    % time series, with options for exclusion zones to avoid trivial matches.
    %
    % SYNTAX:
    %   [knnIndices, distances] = computeKnnDistances(timeSeries, windowSize, k)
    %   [knnIndices, distances] = computeKnnDistances(timeSeries, windowSize, k, 'ExclusionZone', zone)
    %
    % INPUT ARGUMENTS:
    %   timeSeries     - Input time series data (numeric vector)
    %   windowSize     - Size of sliding window (positive integer)
    %   k              - Number of nearest neighbors to find (positive integer)
    %   'ExclusionZone' - Fraction of window size for exclusion zone (default: 0.25)
    %
    % OUTPUT ARGUMENTS:
    %   knnIndices - Matrix of k-nearest neighbor indices for each subsequence
    %                Size: (numSubseq x k)
    %   distances  - Matrix of distances to k-nearest neighbors
    %                Size: (numSubseq x k)
    %
    % ALGORITHM:
    %   1. Extract and z-normalize all subsequences
    %   2. Compute pairwise Euclidean distances
    %   3. Apply exclusion zone around each subsequence
    %   4. Find k smallest distances for each subsequence
    %
    % EXAMPLE:
    %   ts = randn(200, 1);
    %   [indices, dists] = computeKnnDistances(ts, 20, 5);
    %   fprintf('Found %d nearest neighbors for each of %d subsequences\n', ...
    %           size(indices, 2), size(indices, 1));
    %
    % See also: ClaSPTransformer, slidingDotProduct, calculateRocAuc

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'timeSeries', @(x) validateattributes(x, {'numeric'}, {'vector', 'finite'}));
    addRequired(p, 'windowSize', @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
    addRequired(p, 'k', @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
    addParameter(p, 'ExclusionZone', 0.25, @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative'}));
    parse(p, timeSeries, windowSize, k, varargin{:});

    exclusionZoneFraction = p.Results.ExclusionZone;
    timeSeries = timeSeries(:);  % Ensure column vector
    n = length(timeSeries);

    if windowSize >= n
        error('computeKnnDistances:InvalidWindowSize', ...
            'Window size (%d) must be smaller than time series length (%d)', windowSize, n);
    end

    numSubseq = n - windowSize + 1;

    if k >= numSubseq
        error('computeKnnDistances:InvalidK', ...
            'k (%d) must be smaller than number of subsequences (%d)', k, numSubseq);
    end

    % Extract and normalize subsequences
    subsequences = zeros(numSubseq, windowSize);
    for i = 1:numSubseq
        subseq = timeSeries(i:i+windowSize-1);

        % Z-normalize subsequence
        mu = mean(subseq);
        sigma = std(subseq);

        if sigma == 0
            % Handle constant subsequences
            subsequences(i, :) = subseq - mu;
        else
            subsequences(i, :) = (subseq - mu) / sigma;
        end
    end

    % Compute pairwise distances
    if exist('pdist2', 'file') == 2
        % Use optimized pdist2 if Statistics Toolbox is available
        distMatrix = pdist2(subsequences, subsequences);
    else
        % Fallback to manual computation
        distMatrix = computePairwiseDistances(subsequences);
    end

    % Calculate exclusion zone size
    exclusionZoneSize = max(1, round(exclusionZoneFraction * windowSize));

    % Initialize output matrices
    knnIndices = zeros(numSubseq, k);
    distances = zeros(numSubseq, k);

    % Find k-nearest neighbors for each subsequence
    for i = 1:numSubseq
        % Create a copy of distances for this subsequence
        currentDists = distMatrix(i, :);

        % Exclude self-match
        currentDists(i) = inf;

        % Apply exclusion zone to avoid trivial matches
        startExclusion = max(1, i - exclusionZoneSize);
        endExclusion = min(numSubseq, i + exclusionZoneSize);
        currentDists(startExclusion:endExclusion) = inf;

        % Find k smallest distances
        [sortedDists, sortedIndices] = sort(currentDists);
        validIndices = find(~isinf(sortedDists));

        if length(validIndices) >= k
            % Sufficient valid neighbors found
            knnIndices(i, :) = sortedIndices(validIndices(1:k));
            distances(i, :) = sortedDists(validIndices(1:k));
        else
            % Not enough valid neighbors (rare case)
            numValid = length(validIndices);
            if numValid > 0
                knnIndices(i, 1:numValid) = sortedIndices(validIndices);
                distances(i, 1:numValid) = sortedDists(validIndices);

                % Fill remaining slots with the last valid neighbor
                knnIndices(i, numValid+1:k) = sortedIndices(validIndices(end));
                distances(i, numValid+1:k) = sortedDists(validIndices(end));
            else
                % Extremely rare case: no valid neighbors
                % Fill with furthest subsequences
                [~, furthestIndices] = sort(distMatrix(i, :), 'descend');
                validFurthest = furthestIndices(~isinf(distMatrix(i, furthestIndices)));
                if ~isempty(validFurthest)
                    fillIndices = validFurthest(1:min(k, length(validFurthest)));
                    knnIndices(i, 1:length(fillIndices)) = fillIndices;
                    distances(i, 1:length(fillIndices)) = distMatrix(i, fillIndices);
                end
            end
        end
    end
end

function distMatrix = computePairwiseDistances(data)
    % COMPUTEPAIRWISEDISTANCES Manual pairwise distance computation
    %
    % Fallback function when pdist2 is not available

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