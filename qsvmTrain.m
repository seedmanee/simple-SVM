function [A, b, w] = qsvmTrain(X, Y, C, param, Kernel)
% input y = {-1, +1}
% y(i)*y(j)*K(i,j) = y(i) * y(j) * Kernel(i, j),   K: kernel matrix
% m: number of instances

[m, n] = size(X);

y = 2*Y-1;  % {0,1} -> {-1, +1}

eps = 1e-3;
tau = 1e-12;

%disp('precompute start');
%tic
% this is for cache
Q = zeros(m, m);

% special optimization
%if strcmp(func2str(kernel), 'linear')
%  K = X*X';
%elseif strcmp(func2str(kernel), 'polynomial')
%  K = (param*(X*X'+1)).^5;
%elseif strfind(func2str(kernel), 'gaussian')
  X2 = sum(X.^2, 2);
  K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
  K = Kernel(1, 0, param) .^ K;
%end

% ===================

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
%          a = y(i)*y(i)*Kernel(X(i,:),X(i,:),param) + ...
%              y(t)*y(t)*Kernel(X(t,:),X(t,:),param) - ...
%              2*y(i)*y(t)*y(i)*y(t)*Kernel(X(i,:),X(t,:),param);
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

%    a = y(i)*y(i)*Kernel(X(i,:),X(i,:),param) + ...
%        y(j)*y(j)*Kernel(X(j,:),X(j,:),param) - ...
%        2*y(i)*y(j)*y(i)*y(j)*Kernel(X(i,:),X(j,:),param);
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
%      G(t) = G(t) + y(t)*y(i)*Kernel(X(t,:),X(i,:),param)*deltaAi + ...
%             y(t)*y(j)*Kernel(X(t,:),X(j,:),param)*deltaAj;
      G(t) = G(t) + y(t)*y(i)*K(t,i)*deltaAi + y(t)*y(j)*K(t,j)*deltaAj;
    end
  end
  
  w = ((A.*y)'*X)';

%  disp('train done');

end
