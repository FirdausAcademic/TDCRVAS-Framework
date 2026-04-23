function [alpha_traj, lv_traj] = generate_trajectories_inputs(numSec, alpha_bounds, lv_bounds, numPoints)
% GENERATE_TRAJECTORIES Generate trajectory parameters for multiple sections
% Returns trajectories as column matrices
%
% Inputs:
%   numSec     - Number of sections (e.g., 3, 5, etc.)
%   alpha_bounds - Nx2 matrix where each row is [lower, upper] bounds for each section
%   lv_bounds    - Nx2 matrix where each row is [lower, upper] bounds for each section  
%   numPoints  - Number of points in each trajectory (default: 1000)
%
% Outputs:
%   alpha_traj - Matrix where each column is alpha trajectory for a section
%   lv_traj    - Matrix where each column is lv trajectory for a section

    if nargin < 4
        numPoints = 1000;
    end

    % Validate input dimensions
    if size(alpha_bounds, 1) ~= numSec || size(alpha_bounds, 2) ~= 2
        error('alpha_bounds must be a %dx2 matrix with [lower,upper] bounds for each section', numSec);
    end
    
    if size(lv_bounds, 1) ~= numSec || size(lv_bounds, 2) ~= 2
        error('lv_bounds must be a %dx2 matrix with [lower,upper] bounds for each section', numSec);
    end
    
    % Initialize matrices (NOT cell arrays)
    alpha_traj = zeros(numPoints, numSec);
    lv_traj = zeros(numPoints, numSec);
    
    % Generate trajectories for each section with its own bounds
    for i = 1:numSec
        % Get bounds for current section
        alpha_lower = alpha_bounds(i, 1);
        alpha_upper = alpha_bounds(i, 2);
        lv_lower = lv_bounds(i, 1);
        lv_upper = lv_bounds(i, 2);
        
        % Generate trajectories as column vectors and store in matrices
        alpha_traj(:, i) = linspace(alpha_lower, alpha_upper, numPoints)';
        lv_traj(:, i) = linspace(lv_lower, lv_upper, numPoints)';
    end
end