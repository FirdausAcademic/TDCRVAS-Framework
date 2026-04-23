function R_i = computeRotationMatrix(p_i, theta_i)
% computeRotationMatrix  Rodrigues rotation about axis = [1;0;0] × p_i
%
%   R_i = computeRotationMatrix(p_i, theta_i)
%
% Inputs:
%   p_i      – 3×1 position vector [x_i; y_i; z_i]
%   theta_i  – rotation angle (scalar)
%
% Output:
%   R_i      – 3×3 rotation matrix

    % Define the fixed axis
    a = [1; 0; 0];

    % Compute the rotation axis (cross product)
    w = cross(a, p_i);

    % Handle the degenerate case where axis is zero
    norm_w = norm(w);
    if norm_w < eps
        R_i = eye(3);
        return;
    end

    % Unit rotation axis
    k = w / norm_w;

    % Skew‐symmetric matrix of k
    K = [   0    -k(3)   k(2);
          k(3)     0    -k(1);
         -k(2)   k(1)     0   ];

    % Rodrigues’ formula
R_i = cos(theta_i)*eye(3)+ (1 - cos(theta_i))*(k.*k')+ sin(theta_i)*K;

end
