function [model] = qsvmTrain(X, Y, C, param, Kernel)
  % input Y = {0, 1}
  % X: m by n
  % Y: m by 1

  [m, n] = size(X);

  y = 2*Y-1;  % {0,1} -> {-1, +1}

  eps = 1e-3;
  tau = 1e-12;

  % K: kernel matrix

  % this precompute idea is borrowed from Stanford open course: Machine Learning
  % optimization
  if strcmp(Kernel, 'linear')
    K = param*X*X';
  elseif strcmp(Kernel, 'polynomial')
    K = (param*(X*X'+1)).^4;
  elseif strcmp(Kernel, 'gaussian')
    X2 = sum(X.^2, 2);
    K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
    K = gaussian(1, 0, param) .^ K;
  end

  % ===================
  % the training part is based on the paper of libsvm

  A = zeros(m, 1); % alpha array A to all zero
  G = -ones(m, 1); % gradient array G to all -1

  while 1
    %% ================= select working set
    i = -1;
    G_max = -realmax;
    G_min =  realmax;

    for t = 1:m
      if ((y(t) == 1 && A(t) < C) || (y(t) == -1 && A(t) > 0))
        if (-y(t)*G(t) >= G_max)
          i = t;
          G_max = -y(t)*G(t);
        end
      end
    end

    j = -1;
    obj_min = realmax;

    for t = 1:m
      if ((y(t) == 1 && A(t) > 0) || (y(t) == -1 && A(t) < C))
        b = G_max + y(t)*G(t);
        if (-y(t)*G(t) <= G_min)
          G_min = -y(t)*G(t);
        end

        if (b>0)
          a = y(i)*y(i)*K(i,i) + y(t)*y(t)*K(t,t) - 2*y(i)*y(t)*y(i)*y(t)*K(i,t);
          if (a <= 0)
            a = tau;
          end
          if ( -(b*b)/a <= obj_min)
            j = t;
            obj_min = -(b*b)/a;
          end
        end
      end
    end

    if (G_max - G_min < eps)
      break;
    end
    %% end of select working set

    a = y(i)*y(i)*K(i,i) + y(j)*y(j)*K(j,j) - 2*y(i)*y(j)*y(i)*y(j)*K(i,j);

    if ( a <= 0)
      a = tau;
    end

    b = -y(i)*G(i) + y(j)*G(j);

    % update alpha
    oldAi = A(i);
    oldAj = A(j);
    A(i) = A(i) + y(i)*b/a;
    A(j) = A(j) - y(j)*b/a;

    % project alpha back to the feasible region
    tsum = y(i)*oldAi + y(j)*oldAj;
    A(i) = max(min(A(i),C),0);
    A(j) = y(j) * (tsum - y(i)*A(i));
    A(j) = max(min(A(j),C),0);
    A(i) = y(i) * (tsum - y(j)*A(j));

    % update gradient
    deltaAi = A(i) - oldAi;
    deltaAj = A(j) - oldAj;

    for t = 1:m
      G(t) = G(t) + y(t)*y(i)*K(t,i)*deltaAi + y(t)*y(j)*K(t,j)*deltaAj;
    end
  end
  
  % this model struct is copy from Stanford open course: Machine Learning
  idx = A>0;
  model.X= X(idx,:);
  model.y= y(idx);
  model.b = b;
  model.A = A(idx);
  model.Kernel = Kernel;
  model.param = param;
  model.w = ((A.*y)'*X)';

end
