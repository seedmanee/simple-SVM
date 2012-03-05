function [i, j] = selectB(A, G)

  eps = 1e-3;
  tau = 1e-12;

  i = -1;
  G_max = realmin;
  G_min = realmax;

  for t = 1:len
    if ((y(t) == 1 && A(t) < C) || (y(t) == -1 && A(t) > 0))
      if (-y(t)*G(t) >= G_max)
        i = t;
        G_max = -y(t)*G(t);
      end
    end
  end

  j = -1;
  obj_min = realmax;

  for t = 1:len
    if ((y(t) == 1 && A(t) > 0) || (y(t) == -1 && A(t) < C))
      b = G_max + y(t)*G(t);
      if (-y(t)*G(t) <= G_min)
        G_min = -y(t)*G(t);
      end

      if (b>0)
        a = Q(i,i) + Q(t,t) - 2*y(i)*y(t)*Q(i,t);
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
    i = -1;
    j = -1;
  end

end
