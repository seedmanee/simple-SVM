%======================== load file ===========================
clear all;
X = load('pendigits-orig.csv');
Y = X(:, size(X, 2) );
X(:, size(X, 2)) = [];

[m, n] = size(X);

%====================== input test data =========================

% normalize X
X = (X - ones(m, 1) * mean(X)) ./(ones(m,1) * sqrt( var(X)) );

Kernel = @gaussian;
C = 100;
param = 1;

% ===================== k-fold ==================================
kfold_max = 10;
ksize = ceil(m/kfold_max);
error_table = zeros(kfold_max,1);
confusion_matrix = zeros(10, 10, kfold_max);

for kfold_index = 1:kfold_max
  Xhold = X;
  start_pt = (kfold_index - 1) * ksize + 1;
  end_pt = min(start_pt + ksize - 1, m);

  mcv = end_pt - start_pt + 1;
  mhold = m - mcv;

  Xcv = X(start_pt:end_pt, :);
  Ycv = Y(start_pt:end_pt, :);

  Xhold = X;
  Yhold = Y;

  Xhold(start_pt:end_pt,:) = [];
  Yhold(start_pt:end_pt,:) = [];

  Ypredict = zeros(mcv, 1);     % this is predict output

%================== train 10 binary svm =================
  disp(['===== ' int2str(kfold_index) '-fold =====' ]);

  Ksize = 10;

  w = zeros(n, Ksize);
  b = zeros(1, Ksize);

  for ith = 1:Ksize
    [A, b(ith), w(:,ith)] = qsvmTrain(Xhold, (Yhold==(ith-1)), C, param, Kernel);
  end

  % predict, choose output with most confident
  score = Xcv * w + ones(mcv,1)*b;

  [pre_val, pre_idx] = max(score');
  Ypredict = (pre_idx - 1)';

  for y_index = 1: length(Ycv)
    confusion_matrix(Ycv(y_index)+1, Ypredict(y_index)+1, kfold_index) = ...
    confusion_matrix(Ycv(y_index)+1, Ypredict(y_index)+1, kfold_index) + 1;
  end

  error_num = sum(Ycv ~= Ypredict);
  error_table(kfold_index) = error_num/mcv;

  confusion_matrix(:,:,kfold_index)

end % k-fold
