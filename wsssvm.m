% input y = {-1, +1}
% Q(i,j) = y(i) * y(j) * K(i, j),   K: kernel matrix
% len: number of instances

eps = 1e-3;
tau = 1e-12;

A = zeros(m, 1);     % alpha array A to all zero
G = -ones(m, 1); % gradient array G to all -1

while 1
  %% select working set

  %% end of select working set
  (i,j) = selectB();
  if (j == -1)
    break;
  end

  a = Q(i,i) + Q(j,j) - 2*y(i)*y(j)*Q(i,j);
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
  sum = y(i)*oldAi + y(j)*oldAj;
  A(i) = max(min(A(i),C),0);
  A(j) = y(j) * (sum - y(i)*A(i));
  A(j) = max(min(A(j),C),0);
  A(i) = y(i) * (sum - y(j)*A(j));

  % update gradient
  deltaAi = A(i) - oldAi;
  deltaAj = A(j) - oldAj;

  for t = 1:len
    G(t) = G(t) + Q(t,i)*deltaAi + Q(t,j)*deltaAj;
  end
end
