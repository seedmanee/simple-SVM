function [pred] = qsvmPredict (X, b, w)
  score = zeros(10, 1);

  for idx = 1:10
    score = w(:,idx)' * X + b(idx);
  end

  [x, ix] = max(score);
  pred = ix + 1;
