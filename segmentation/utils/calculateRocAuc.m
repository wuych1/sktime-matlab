function auc = calculateRocAuc(scores, labels)
    % CALCULATERAUCAUC Compute Area Under the ROC Curve
    %
    % This function calculates the Area Under the Receiver Operating
    % Characteristic (ROC) Curve for binary classification performance.
    %
    % SYNTAX:
    %   auc = calculateRocAuc(scores, labels)
    %
    % INPUT ARGUMENTS:
    %   scores - Prediction scores or probabilities (numeric vector)
    %   labels - True binary labels (logical or numeric vector with values 0/1)
    %
    % OUTPUT ARGUMENTS:
    %   auc - Area under ROC curve (scalar in range [0, 1])
    %         0.5 indicates random performance
    %         1.0 indicates perfect classification
    %         0.0 indicates perfect inverse classification
    %
    % ALGORITHM:
    %   1. Sort scores in descending order
    %   2. Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    %   3. Compute AUC using trapezoidal integration
    %
    % NOTES:
    %   - Handles edge cases (single class, empty inputs)
    %   - Uses efficient vectorized computation
    %   - Returns 0.5 for degenerate cases (random performance)
    %
    % EXAMPLE:
    %   % Generate synthetic data
    %   scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1];
    %   labels = [1, 1, 1, 0, 0, 0, 1, 0];
    %   auc = calculateRocAuc(scores, labels);
    %   fprintf('AUC: %.3f\n', auc);
    %
    % REFERENCES:
    %   [1] Fawcett, T. (2006). An introduction to ROC analysis.
    %       Pattern Recognition Letters, 27(8), 861-874.
    %
    % See also: ClaSPTransformer, computeKnnDistances

    % Input validation
    validateattributes(scores, {'numeric'}, {'vector', 'finite'});
    validateattributes(labels, {'numeric', 'logical'}, {'vector'});

    scores = scores(:);  % Ensure column vector
    labels = logical(labels(:));  % Convert to logical column vector

    if length(scores) ~= length(labels)
        error('calculateRocAuc:DimensionMismatch', ...
            'Scores and labels must have the same length');
    end

    % Handle empty input
    if isempty(scores)
        auc = 0.5;
        return;
    end

    % Handle single class cases
    uniqueLabels = unique(labels);
    if length(uniqueLabels) < 2
        auc = 0.5;  % Random performance for single class
        return;
    end

    % Count positive and negative samples
    numPos = sum(labels);
    numNeg = length(labels) - numPos;

    % Handle degenerate cases
    if numPos == 0 || numNeg == 0
        auc = 0.5;
        return;
    end

    % Sort scores in descending order
    [sortedScores, sortOrder] = sort(scores, 'descend');
    sortedLabels = labels(sortOrder);

    % Calculate cumulative true positives and false positives
    cumulativeTP = cumsum(sortedLabels);
    cumulativeFP = cumsum(~sortedLabels);

    % Calculate True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
    tpr = cumulativeTP / numPos;
    fpr = cumulativeFP / numNeg;

    % Add the origin point (0, 0) to complete the ROC curve
    tpr = [0; tpr];
    fpr = [0; fpr];

    % Handle duplicate scores by removing redundant points
    % This ensures proper AUC calculation when multiple samples have identical scores
    [uniqueFpr, uniqueIdx] = unique(fpr, 'first');
    uniqueTpr = tpr(uniqueIdx);

    % Calculate AUC using trapezoidal integration
    % AUC = ∫ TPR d(FPR) from FPR=0 to FPR=1
    if length(uniqueFpr) > 1
        auc = trapz(uniqueFpr, uniqueTpr);
    else
        % Edge case: all FPR values are the same
        auc = 0.5;
    end

    % Ensure AUC is within valid range [0, 1]
    auc = max(0, min(1, auc));

    % Handle numerical precision issues
    if abs(auc - 0.5) < 1e-10
        auc = 0.5;
    elseif abs(auc - 1.0) < 1e-10
        auc = 1.0;
    elseif abs(auc) < 1e-10
        auc = 0.0;
    end
end