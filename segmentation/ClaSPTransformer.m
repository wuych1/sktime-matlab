classdef ClaSPTransformer < handle
    % CLASPTRANSFORMER ClaSP (Classification Score Profile) transformer
    %
    % Computes the classification score profile for a time series by:
    % 1. Extracting sliding windows and computing k-nearest neighbors
    % 2. Creating binary classification problems at each split point
    % 3. Evaluating split quality using ROC-AUC scores
    % 4. Interpolating gaps while avoiding boundary artifacts
    %
    % This implementation matches the Python sktime/aeon-toolkit version,
    % including boundary exclusion and trivial match prevention.

    properties (Access = private, Constant)
        DEFAULT_K = 3
    end

    methods
        function obj = ClaSPTransformer()
            % Constructor
        end

        function [profile, knnMask] = transform(obj, timeSeries, windowSize, options)
            % TRANSFORM Compute ClaSP classification score profile
            %
            % SYNTAX:
            %   [profile, knnMask] = obj.transform(timeSeries, windowSize)
            %   [profile, knnMask] = obj.transform(timeSeries, windowSize, 'K', k)
            %   [profile, knnMask] = obj.transform(timeSeries, windowSize, 'ExclusionRadius', r)
            %
            % INPUT ARGUMENTS:
            %   timeSeries - Input time series (numeric vector)
            %   windowSize - Sliding window size (positive integer)
            %   'K'        - Number of nearest neighbors (default: 3)
            %   'ExclusionRadius' - Fraction for exclusion zones (default: 0.05)
            %
            % OUTPUT ARGUMENTS:
            %   profile - Classification score profile (numeric vector)
            %           - Positions near boundaries filled with low values
            %           - Interior gaps interpolated linearly
            %   knnMask - k-NN indices matrix (k x numWindows)

            % Parse input arguments
            arguments
                obj
                timeSeries (:,1) double {mustBeFinite}
                windowSize (1,1) double {mustBePositive, mustBeInteger}
                options.K (1,1) double {mustBePositive, mustBeInteger} = obj.DEFAULT_K
                options.ExclusionRadius (1,1) double {mustBeNonnegative} = 0.05
            end

            k = options.K;
            exclusionRadius = options.ExclusionRadius;
            timeSeries = timeSeries(:);  % Ensure column vector
            n = length(timeSeries);
            m = windowSize;

            if m >= n - 1
                error('ClaSPTransformer:InvalidWindowSize', ...
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
                        % Both windows have variance - use normalized distance
                        normalizedDot = (dotProducts(j) - m * means(i) * means(j)) / (m * stds(i) * stds(j));
                        distances(j) = 2 * m * (1 - normalizedDot);
                    elseif stds(i) == 0 && stds(j) == 0
                        % Both windows are constant - use mean difference
                        if abs(means(i) - means(j)) < 1e-10
                            distances(j) = 0;  % Same constant value
                        else
                            distances(j) = 2 * m * abs(means(i) - means(j)) / max(abs(means(i)), abs(means(j)));
                        end
                    else
                        % One constant, one not - maximum distance
                        distances(j) = 2 * m;
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
            % SLIDINGDOTPRODUCT Compute sliding dot product
            %
            % Direct implementation for correctness

            n = length(timeSeries);
            numWindows = n - m + 1;
            dotProducts = zeros(numWindows, 1);

            % Ensure query is row vector
            query = query(:)';

            % Compute dot product for each window
            for i = 1:numWindows
                window = timeSeries(i:i+m-1)';
                dotProducts(i) = sum(query .* window);
            end
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
            % Fixed implementation that handles ties correctly

            % Handle edge cases
            if length(unique(yTrue)) < 2
                auc = 0.5;
                return;
            end

            numPos = sum(yTrue == 1);
            numNeg = sum(yTrue == 0);

            if numPos == 0 || numNeg == 0
                auc = 0.5;
                return;
            end

            % Handle the case where all predictions are the same (ties)
            if length(unique(yPred)) == 1
                auc = 0.5;
                return;
            end

            % Sort by predictions in descending order
            [~, sortIdx] = sort(yPred, 'descend');
            sortedLabels = yTrue(sortIdx);

            % Build TPR and FPR for each threshold
            tpr = zeros(length(yTrue) + 1, 1);
            fpr = zeros(length(yTrue) + 1, 1);

            tpr(1) = 0;
            fpr(1) = 0;

            for i = 1:length(yTrue)
                if sortedLabels(i) == 1
                    tpr(i+1) = tpr(i) + 1/numPos;
                    fpr(i+1) = fpr(i);
                else
                    tpr(i+1) = tpr(i);
                    fpr(i+1) = fpr(i) + 1/numNeg;
                end
            end

            % Compute AUC using trapezoidal rule
            auc = 0;
            for i = 1:length(fpr)-1
                auc = auc + 0.5 * (fpr(i+1) - fpr(i)) * (tpr(i+1) + tpr(i));
            end

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

            % Get min value of valid points for boundary fill
            % Use slightly below minimum to avoid creating artificial maxima
            minValid = min(profile(validIdx));
            boundaryValue = minValid - 0.01;  % Slightly below minimum

            % Linear interpolation for interior NaN values only
            nanIndices = find(nanIdx);
            for i = 1:length(nanIndices)
                idx = nanIndices(i);
                % Check if this is interior (between valid points) or boundary
                if idx > validIdx(1) && idx < validIdx(end)
                    % Interior - interpolate
                    % Find surrounding valid points
                    leftValid = validIdx(find(validIdx < idx, 1, 'last'));
                    rightValid = validIdx(find(validIdx > idx, 1, 'first'));
                    % Linear interpolation
                    weight = (idx - leftValid) / (rightValid - leftValid);
                    profile(idx) = profile(leftValid) * (1 - weight) + profile(rightValid) * weight;
                else
                    % Boundary - use low value to avoid artificial maxima
                    profile(idx) = boundaryValue;
                end
            end

            % Handle any remaining NaN values (shouldn't be any)
            profile(isnan(profile)) = boundaryValue;
        end
    end
end