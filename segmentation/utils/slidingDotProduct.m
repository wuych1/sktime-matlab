function dotProducts = slidingDotProduct(timeSeries, windowSize)
    % SLIDINGDOTPRODUCT Efficient computation of sliding window dot products
    %
    % This function computes dot products between all pairs of subsequences
    % in a time series using a sliding window approach with z-normalization.
    %
    % SYNTAX:
    %   dotProducts = slidingDotProduct(timeSeries, windowSize)
    %
    % INPUT ARGUMENTS:
    %   timeSeries - Input time series data (numeric vector)
    %   windowSize - Size of sliding window (positive integer)
    %
    % OUTPUT ARGUMENTS:
    %   dotProducts - Matrix of normalized dot products between subsequences
    %                 Size: (numSubseq x numSubseq) where numSubseq = length(timeSeries) - windowSize + 1
    %
    % ALGORITHM:
    %   1. Extract all subsequences of specified window size
    %   2. Apply z-normalization to each subsequence
    %   3. Compute dot products between all pairs of normalized subsequences
    %
    % EXAMPLE:
    %   ts = randn(100, 1);
    %   dotProd = slidingDotProduct(ts, 10);
    %   imagesc(dotProd);  % Visualize similarity matrix
    %
    % See also: ClaSPTransformer, computeKnnDistances

    % Input validation
    validateattributes(timeSeries, {'numeric'}, {'vector', 'finite'});
    validateattributes(windowSize, {'numeric'}, {'scalar', 'positive', 'integer'});

    timeSeries = timeSeries(:);  % Ensure column vector
    n = length(timeSeries);

    if windowSize >= n
        error('slidingDotProduct:InvalidWindowSize', ...
            'Window size (%d) must be smaller than time series length (%d)', windowSize, n);
    end

    numSubseq = n - windowSize + 1;

    % Preallocate subsequence matrix
    subsequences = zeros(windowSize, numSubseq);

    % Extract all subsequences
    for i = 1:numSubseq
        subsequences(:, i) = timeSeries(i:i+windowSize-1);
    end

    % Compute means and standard deviations for z-normalization
    means = mean(subsequences, 1);
    stds = std(subsequences, 0, 1);

    % Handle constant subsequences (std = 0) to avoid division by zero
    stds(stds == 0) = 1;

    % Apply z-normalization
    normalizedSubseq = (subsequences - means) ./ stds;

    % Compute dot products between all pairs of normalized subsequences
    dotProducts = normalizedSubseq' * normalizedSubseq;

    % Ensure diagonal elements are exactly 1 (self-similarity)
    dotProducts(1:numSubseq+1:end) = 1;
end