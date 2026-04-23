function [p_tip, R_tip,H_all] = forwardKinematicsMultiSection(Ls, rd, lv, alpha)
% forwardKinematicsMultiSection  Compute tip pose of an n‐section continuum robot
% using forwardKinematicsSingleSection, computeRotationMatrix, and makeHomogeneousTransform.
%
%   [p_tip, R_tip] = forwardKinematicsMultiSection(Ls, rd, lv, alpha)
%
% Inputs:
%   Ls    – 1×nSec vector of undeformed arc‐lengths [L1, …, Ln]
%   rd    – scalar disk‐radius parameter
%   lv    – 1×nSec vector of cable extensions [lv1, …, lvn]
%   alpha – 1×nSec vector of bending‐plane angles [α1, …, αn]
%
% Outputs:
%   p_tip – 3×1 position of the final tip in the base frame
%   R_tip – 3×3 orientation (rotation matrix) of the final tip in the base frame

nSec = numel(Ls);
H_all = zeros(4,4,nSec);    % preallocate storage for each H_i
H = eye(4);                 % overall transform

% loop through each section
for i = 1:nSec
    % 1) local tip position p_i
    p_i = forwardKinematicsSingleSection(Ls(i), rd, lv(i), alpha(i));

    % 2) bending angle theta_i
    theta_i = lv(i) / rd;

    % 3) rotation about axis [1;0;0] × p_i
    R_i = computeRotationMatrix(p_i, theta_i); % some issue was ditected
    %in this calculation
    %R_i = rotation_matrix_direct_formula(p_i, theta_i);

    % 4) build homogeneous transform H_i
    H_i = makeHomogeneousTransform(R_i, p_i);
    H_all(:,:,i) = H_i;     % save this section’s H_i
    % 5) accumulate into global transform
    H = H * H_i;
end

% extract final tip pose
p_tip = H(1:3,4);
R_tip = H(1:3,1:3);
end
