classdef ClaSPTransformerExact < handle
    % CLASPTRANSFORMEREXACT Exact implementation of ClaSP transformer
    %
    % This implementation follows the exact algorithm specification from the
    % Python reference implementation in aeon-toolkit.

    properties (Access = private, Constant)
        DEFAULT_K = 3
    end

    methods
        function obj = ClaSPTransformerExact()
            % Constructor
        end

        function [profile, knnMask] = transform(obj, timeSeries, windowSize, varargin)
            % TRANSFORM Compute ClaSP classification score profile
            %
            % SYNTAX:
            %   [profile, knnMask] = obj.transform(timeSeries, windowSize)
            %   [profile, knnMask] = obj.transform(timeSeries, windowSize, 'K', k)
            %
            % INPUT ARGUMENTS:
            %   timeSeries - Input time series (numeric vector)
            %   windowSize - Sliding window size (positive integer)
            %   'K'        - Number of nearest neighbors (default: 3)
            %
            % OUTPUT ARGUMENTS:
            %   profile - Classification score profile (numeric vector)
            %   knnMask - k-NN indices matrix (k x numWindows)

            % Parse input arguments
            p = inputParser;
            addRequired(p, 'timeSeries', @(x) validateattributes(x, {'numeric'}, {'vector'}));
            addRequired(p, 'windowSize', @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            addParameter(p, 'K', obj.DEFAULT_K, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
            addParameter(p, 'ExclusionRadius', 0.05, @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative'}));
            parse(p, timeSeries, windowSize, varargin{:});

            k = p.Results.K;
            exclusionRadius = p.Results.ExclusionRadius;
            timeSeries = timeSeries(:);  % Ensure column vector
            n = length(timeSeries);
            m = windowSize;

            if m >= n - 1
                error('ClaSPTransformerExact:InvalidWindowSize', ...
                    'Window size (%d) must be smaller than time series length (%d)', m, n-1);
            end

            numWindows = n - m + 1;

            % Step 1: Compute k-nearest neighbors
            fprintf('Computing k-nearest neighbors...\n');
            knnMask = obj.computeDistancesIterative(timeSeries, m, k);

            % Step 2: Compute classification profile
            fprintf('Computing classification profile...\n');
            exclusionZone = max(m, round(numWindows * exclusionRadius));
            profile = obj.calcProfile(m, knnMask, exclusionZone);

            % Step 3: Interpolate to fill NaN values
            fprintf('Interpolating profile...\n');
            profile = obj.interpolateProfile(profile);
        end

        function knnMask = computeDistancesIterative(obj, timeSeries, m, k)
            % COMPUTEDISTANCESITERATIVE Find k-nearest neighbors for each window
            %
            % This follows the exact Python implementation

            n = length(timeSeries);
            numWindows = n - m + 1;

            % Step 1: Extract sliding windows
            windows = obj.slidingWindow(timeSeries, m);

            % Step 2: Compute means and standard deviations
            means = mean(windows, 2);
            stds = std(windows, 0, 2);
            stds(stds == 0) = 1;  % Avoid division by zero

            % Step 3: Compute distance matrix using sliding dot products
            knnMask = zeros(k, numWindows);
            exclusionZone = max(1, round(m * 0.5));  % 50% of window size (Python default)

            for i = 1:numWindows
                % Compute sliding dot product for current window
                query = windows(i, :);
                dotProducts = obj.slidingDotProduct(query, timeSeries, m);

                % Compute normalized distances
                distances = zeros(1, numWindows);
                for j = 1:numWindows
                    if stds(i) > 0 && stds(j) > 0
                        normalizedDot = (dotProducts(j) - m * means(i) * means(j)) / (m * stds(i) * stds(j));
                        distances(j) = 2 * m * (1 - normalizedDot);
                    else
                        distances(j) = 2 * m;  % Maximum distance for constant windows
                    end
                end

                % Apply exclusion zone
                distances(i) = inf;  % Exclude self
                startExcl = max(1, i - exclusionZone);
                endExcl = min(numWindows, i + exclusionZone);
                distances(startExcl:endExcl) = inf;

                % Find k nearest neighbors
                [~, sortedIdx] = sort(distances);
                validIdx = sortedIdx(~isinf(distances(sortedIdx)));

                if length(validIdx) >= k
                    knnMask(:, i) = validIdx(1:k);
                else
                    % Handle case with insufficient neighbors
                    numValid = length(validIdx);
                    if numValid > 0
                        knnMask(1:numValid, i) = validIdx;
                        knnMask(numValid+1:k, i) = validIdx(end);  % Repeat last
                    else
                        knnMask(:, i) = 1;  % Fallback
                    end
                end
            end
        end

        function windows = slidingWindow(obj, timeSeries, m)
            % SLIDINGWINDOW Extract sliding windows from time series
            %
            % Equivalent to numpy stride_tricks

            n = length(timeSeries);
            numWindows = n - m + 1;
            windows = zeros(numWindows, m);

            for i = 1:numWindows
                windows(i, :) = timeSeries(i:i+m-1);
            end
        end

        function dotProducts = slidingDotProduct(obj, query, timeSeries, m)
            % SLIDINGDOTPRODUCT Compute sliding dot product using FFT
            %
            % This follows the exact Python implementation using FFT

            n = length(timeSeries);
            numWindows = n - m + 1;

            % Handle odd length (Python implementation detail)
            timeSeriesAdd = 0;
            if mod(n, 2) == 1
                timeSeries = [0; timeSeries];
                timeSeriesAdd = 1;
            end

            % Reverse query and pad
            query = query(end:-1:1);  % Reverse
            query = [query, zeros(1, n - m + timeSeriesAdd)];

            % Compute FFT-based convolution
            fftSeries = fft(timeSeries);
            fftQuery = fft(query);
            result = ifft(fftSeries .* fftQuery');

            % Extract relevant portion
            dotProducts = real(result(m + timeSeriesAdd:end));
            dotProducts = dotProducts(1:numWindows);  % Trim to correct length
        end

        function profile = calcProfile(obj, m, knnMask, exclusionZone)
            % CALCPROFILE Compute classification score profile
            %
            % This follows the exact Python implementation

            [k, numWindows] = size(knnMask);
            profile = nan(1, numWindows);

            % Iterate through potential split points
            for splitIdx = exclusionZone+1:numWindows-exclusionZone
                % Step 1: Create binary labels (exact Python logic)
                % Python: np.zeros(split_idx) + np.ones(n_timepoints - split_idx)
                yTrue = [zeros(1, splitIdx), ones(1, numWindows-splitIdx)];

                % Step 2: Generate k-NN predictions using exact Python logic
                [yTrue, yPred] = obj.calcKnnLabels(knnMask, splitIdx, m, yTrue);

                % Step 3: Compute ROC-AUC score
                profile(splitIdx) = obj.rocAucScore(yTrue, yPred);
            end
        end

        function [yTrue, yPred] = calcKnnLabels(obj, knnMask, splitIdx, m, yTrue)
            % CALCKNNLABELS Generate k-NN predictions for split point
            %
            % This follows the exact Python implementation

            [k, numWindows] = size(knnMask);

            % Create k-NN mask labels
            knnMaskLabels = zeros(k, numWindows);
            for iNeighbor = 1:k
                neighbors = knnMask(iNeighbor, :);
                knnMaskLabels(iNeighbor, :) = yTrue(neighbors);
            end

            % Compute k-NN prediction using majority vote
            onesCount = sum(knnMaskLabels, 1);
            zerosCount = k - onesCount;
            yPred = double(onesCount > zerosCount);

            % Apply exclusion zone at split point (exact Python logic)
            % Python: exclusion_zone = np.arange(split_idx - m, split_idx)
            exclusionStart = max(1, splitIdx - m);
            exclusionEnd = min(numWindows, splitIdx - 1);
            if exclusionStart <= exclusionEnd
                yPred(exclusionStart:exclusionEnd) = 1;  % Force to class 1 (right side)
            end
        end

        function auc = rocAucScore(obj, yTrue, yPred)
            % ROCAUCSCORE Compute ROC-AUC score
            %
            % This follows the exact Python sklearn implementation

            % Handle edge cases
            if length(unique(yTrue)) < 2
                auc = 0.5;
                return;
            end

            % Convert to logical for easier handling
            yTrue = logical(yTrue);

            % Sort by prediction scores (descending)
            [~, sortIdx] = sort(yPred, 'descend');
            sortedLabels = yTrue(sortIdx);

            % Compute counts
            numPos = sum(yTrue);
            numNeg = length(yTrue) - numPos;

            if numPos == 0 || numNeg == 0
                auc = 0.5;
                return;
            end

            % Compute cumulative TP and FP
            cumulativeTP = cumsum(sortedLabels(:));
            cumulativeFP = cumsum(~sortedLabels(:));

            % Compute TPR and FPR
            tpr = [0; cumulativeTP / numPos];
            fpr = [0; cumulativeFP / numNeg];

            % Compute AUC using trapezoidal rule
            auc = trapz(fpr, tpr);

            % Ensure AUC is in [0, 1]
            auc = max(0, min(1, auc));
        end

        function profile = interpolateProfile(obj, profile)
            % INTERPOLATEPROFILE Fill NaN values in profile
            %
            % This mimics pandas interpolate(limit_direction="both")

            nanIdx = isnan(profile);

            if ~any(nanIdx)
                return;  % No NaN values
            end

            % Find valid (non-NaN) indices
            validIdx = find(~nanIdx);

            if length(validIdx) < 2
                % Not enough valid points for interpolation
                profile(nanIdx) = 0.5;  % Default value
                return;
            end

            % Linear interpolation for interior NaN values
            profile(nanIdx) = interp1(validIdx, profile(validIdx), find(nanIdx), 'linear', 'extrap');

            % Handle any remaining NaN values
            profile(isnan(profile)) = 0.5;
        end
    end
end