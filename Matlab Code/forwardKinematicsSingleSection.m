function p_i = forwardKinematicsSingleSection(Li, rd, li, alpha_i)
% forwardKinematicsSingleSection  Compute the local tip position p_i
%
%   p_i = forwardKinematicsSingleSection(Li, rd, li, alpha_i)
%
% Inputs:
%   Li      – nominal (undeformed) arc‐length of section i
%   rd      – disk‐radius parameter
%   li      – current cable extension Δℓ_i
%   alpha_i – bending‐plane angle α_i
%
% Output:
%   p_i     – 3×1 tip‐position vector in the local frame

    % compute bending angle
    theta_i = li / rd;

    % compute radius of curvature
    R_i = Li * rd / li;

    % forward‐kinematics formula
    p_i = [ ...
        R_i * sin(theta_i); ...
        R_i * (1 - cos(theta_i)) * cos(alpha_i); ...
        R_i * (1 - cos(theta_i)) * sin(alpha_i) ...
    ];
end
