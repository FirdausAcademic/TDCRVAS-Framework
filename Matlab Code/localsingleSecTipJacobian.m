function J = localsingleSecTipJacobian(Li, rd, lv, alpha)
% localTipJacobian  Compute the 3×2 Jacobian of the tip of section i
%
%   J = localTipJacobian(Li, rd, lv, alpha)
%
% Inputs:
%   Li    – nominal (undeformed) arc‐length of section i
%   rd    – disk‐radius parameter
%   lv    – current cable extension Δℓ_i
%   alpha – bending‐plane angle α_i
%
% Output:
%   J  – 3×2 Jacobian matrix s.t. ẋ = J * [ėlv; ėalpha]

    %--- compute intermediate variables ---%
    theta = lv/rd;        % θ_i = lvi / rd
    Ri    = Li*rd/lv;     % R_i = L_i * rd / lv

    %--- common term for the (2,1) and (3,1) entries ---%
    common = -Ri/lv*(1 - cos(theta)) + (Ri/rd)*sin(theta);

    %--- assemble Jacobian ---%
    J = zeros(3,2);
    % ∂p/∂lv
    J(1,1) = -Ri/lv * sin(theta)   + (Ri/rd)*cos(theta);
    J(2,1) = common * cos(alpha);
    J(3,1) = common * sin(alpha);

    % ∂p/∂α
    J(1,2) = 0;
    J(2,2) = -Ri*(1 - cos(theta)) * sin(alpha);
    J(3,2) =  Ri*(1 - cos(theta)) * cos(alpha);
%     dp_dlv=J(:,1);
%     dp_dalpha=J(:,2);
end
