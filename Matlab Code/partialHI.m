function [dH_dlv, dH_dalpha] = partialHI(Li, rd, lv, alpha)
% partialHI  Compute ∂H_i/∂lv and ∂H_i/∂alpha for one continuum‐section
%
%   [dH_dlv, dH_dalpha] = partialHI(Li, rd, lv, alpha)
%
% Uses your existing localsingleSecTipJacobian.m to get J_i = [∂p/∂lv, ∂p/∂alpha].
%
% Inputs:
%   Li    – undeformed arc‐length of section i
%   rd    – disk‐radius parameter
%   lv    – current cable extension Δℓ_i
%   alpha – bending‐plane angle α_i
%
% Outputs:
%   dH_dlv    – 4×4 matrix ∂H_i/∂lv
%   dH_dalpha – 4×4 matrix ∂H_i/∂alpha

    % fixed world axis
    a = [1;0;0];

    % 1) tip position and Jacobian
    p_i = forwardKinematicsSingleSection(Li, rd, lv, alpha); 
    J   = localsingleSecTipJacobian(Li, rd, lv, alpha);
    dp_dlv    = J(:,1);
    dp_dalpha = J(:,2);

    % 2) build rotation axis and unit axis
    w = cross(a, p_i);
    nw = norm(w);
    if nw < eps
        k = [1;0;0];
    else
        k = w / nw;
    end

    % 3) Rodrigues parameters
    theta = lv/rd;
    dtheta_dlv = 1/rd;

    K = [   0   -k(3)   k(2);
          k(3)    0    -k(1);
         -k(2)  k(1)     0   ];

    % 4) ∂R/∂θ (treating k constant)
    dR_dtheta = -sin(theta)*eye(3) ...
                + sin(theta)*(k*k.') ...
                + cos(theta)*K;

    % 5) ∂w/∂lv and ∂w/∂alpha
    dw_dlv    = cross(a, dp_dlv);
    dw_dalpha = cross(a, dp_dalpha);

    % 6) ∂k/∂lv and ∂k/∂alpha
    if nw < eps
        dk_dlv    = zeros(3,1);
        dk_dalpha = zeros(3,1);
    else
        P = eye(3) - k*k.';
        dk_dlv    = (P * dw_dlv)    / nw;
        dk_dalpha = (P * dw_dalpha) / nw;
    end

    % 7) precompute ∂R/∂k_j for j=1..3
    for j = 1:3
        e = zeros(3,1); e(j) = 1;
        dR_dkj(:,:,j) = (1-cos(theta))*(e*k.' + k*e.') ...
                        + sin(theta)*[   0   -e(3)  e(2);
                                      e(3)    0   -e(1);
                                     -e(2)  e(1)    0   ];
    end

    % 8) assemble ∂R/∂lv and ∂R/∂alpha
    dR_dlv    = dR_dtheta * dtheta_dlv;
    dR_dalpha = zeros(3,3);
    for j = 1:3
        dR_dlv    = dR_dlv    + dR_dkj(:,:,j) * dk_dlv(j);
        dR_dalpha = dR_dalpha + dR_dkj(:,:,j) * dk_dalpha(j);
    end

    % 9) build the 4×4 partial‐transform matrices
    dH_dlv    = [ dR_dlv,    dp_dlv;
                  0 0 0,     0      ];
    dH_dalpha = [ dR_dalpha, dp_dalpha;
                  0 0 0,     0      ];
end
