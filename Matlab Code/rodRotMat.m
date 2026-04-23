%% Rodrigues helper
function R = rodRotMat(k, theta)
  k = k / norm(k);
  K = [    0   -k(3)  k(2);
        k(3)    0   -k(1);
       -k(2)  k(1)     0 ];
  R = eye(3)*cos(theta) + (1-cos(theta))*(k*k.') + sin(theta)*K;
  %R = eye(3)*cos(theta) + (1-cos(theta))*(k*k.') + sin(theta)*K;
end
