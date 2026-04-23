function varargout = forwardKinematicsCC(L, r_d, n,  del_lv,alpha)
% FORWARDKINEMATICSCC  Continuum‐link kinematics (tip and optional full curve)
%
%  tip = forwardKinematicsCC(L, r_d, n, alpha, del_lv)
%    → returns 3×1 tip position [x;y;z]
%
%  [tip, theta] = forwardKinematicsCC(L, r_d, n, alpha, del_lv)
%    → returns tip and total bend angle (rad)
%
%  [curvePts, tip, theta] = forwardKinematicsCC(L, r_d, n, alpha, del_lv)
%    → returns M×3 curve points, tip, and bend angle
%
% Inputs:
%   L       total backbone length
%   r_d     disk radius
%   n       number of disks
%   alpha   bending‐plane angle w.r.t. y‐axis [rad]
%   del_lv  change in virtual cable length [m]
%
% See also: nargout, varargout

% 1) common preprocessing
% Element‐wise increments
del_lv = abs(del_lv);
del_lv_elem = del_lv / (n - 1);
L_elem      = L      / (n - 1);

% Radius of curvature
R = (L_elem * r_d) / del_lv_elem;

% Total bending angle
theta_bar = del_lv_elem / r_d;
theta     = theta_bar * (n - 1);
% 2) tip coordinate
x_tip = R * sin(theta);
rho   = R - R * cos(theta);
y_tip = rho * cos(alpha);
z_tip = rho * sin(alpha);
tip   = [x_tip; y_tip; z_tip];

% 3) dispatch outputs
switch nargout
    case 1
        % only tip
        varargout{1} = tip;

    case 2
        % tip + bend angle
        varargout{1} = tip;
        varargout{2} = theta;

    case 3
        % Marker density along the curve
        FCM_markerDensity = 150;
        % Generate equally spaced angle samples from 0 to theta
        theta_i = linspace(0, theta, FCM_markerDensity + 1)';

        % Precompute cos(alpha) and sin(alpha)
        cA = cos(alpha);
        sA = sin(alpha);

        % Vectorized curve computation
        x = R * sin(theta_i);
        rho = R - R * cos(theta_i);
        y = rho * cA;
        z = rho * sA;
        curvePts = [x, y, z];

        varargout{1} = curvePts;
        varargout{2} = tip;
        varargout{3} = theta;

    otherwise
        error("forwardKinematicsCC:InvalidOutputCount", ...
            "You requested %d outputs but only 1–3 are supported.", nargout);
end
end
