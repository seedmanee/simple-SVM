% this part is basically borrowed from Stanford open course: Machine Learning

function [pred p]= svmPredict(model, X)
  % pred is prediction {0, 1}
  % p is margin 

  m = size(X, 1);
  p = zeros(m, 1);
  pred = zeros(m, 1);

  if strcmp(model.Kernel, 'linear')
      p = model.param * X * model.w + model.b;
  elseif strcmp(model.Kernel, 'polynomial')
      K = (model.param*(X * model.X' + 1)).^4;
      K = bsxfun(@times, model.y', K);
      K = bsxfun(@times, model.A', K);
      p = sum(K, 2);
  elseif strcmp(model.Kernel, 'gaussian')
      X1 = sum(X.^2, 2);
      X2 = sum(model.X.^2, 2)';
      K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
      K = gaussian(1, 0, model.param) .^ K;
      K = bsxfun(@times, model.y', K);
      K = bsxfun(@times, model.A', K);
      p = sum(K, 2);
  end

  % Convert predictions into 0 / 1
  pred(p >= 0) =  1;
  pred(p <  0) =  0;

end
