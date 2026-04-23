function J = genMultiSecJacobianFixedBase(Ls, rd, lv, alpha)
% genMultiSecJacobianFixedBase  Compute the 3×(2n) Jacobian for an n-section continuum robot
% with the base fixed.
%
%   J = genMultiSecJacobianFixedBase(Ls, rd, lv, alpha)
%
% Inputs:
%   Ls    – 1×n vector of undeformed arc-lengths [L1,…,Ln]
%   rd    – scalar disk-radius parameter
%   lv    – 1×n vector of cable extensions Δℓ_i
%   alpha – 1×n vector of bending-plane angles α_i
%
% Output:
%   J     – 3×(2n) Jacobian mapping [dotℓ; dotα] → dot p_tip (in base frame)

n = numel(Ls);

%% 1) Get each section’s H_i and the tip in the (fixed) base frame
[~, ~, H_all] = forwardKinematicsMultiSection(Ls, rd, lv, alpha);

%% 2) Build prefix and suffix chains **starting** from identity
P = repmat(eye(4),1,1,n+1);
for i = 1:n
  P(:,:,i+1) = P(:,:,i) * H_all(:,:,i);
end

S = repmat(eye(4),1,1,n+2);
for i = n:-1:1
  S(:,:,i) = H_all(:,:,i) * S(:,:,i+1);
end

%% 3) Shape-motion Jacobian: two columns per section
J = zeros(3,2*n);

% sections 1..n-1: full ∂H_i/∂ℓ_i, ∂H_i/∂α_i
for i = 1:n-1
  [dH_dlv, dH_dalpha] = partialHI(Ls(i), rd, lv(i), alpha(i));
  T_l     = P(:,:,i) * dH_dlv    * S(:,:,i+1);
  T_alpha = P(:,:,i) * dH_dalpha * S(:,:,i+1);

  J(:,2*i-1) = T_l(1:3,4);
  J(:,2*i)   = T_alpha(1:3,4);
end

% section n: only ∂p_n/∂ℓ_n, ∂p_n/∂α_n (no ∂R_n terms)
Jn           = localsingleSecTipJacobian(Ls(n), rd, lv(n), alpha(n));
dp_dlv_n     = Jn(:,1);
dp_dalpha_n  = Jn(:,2);

% the rotation prefix R_prefix = H0*H1*…*H_{n-1}, but H0=I so
R_prefix = P(1:3,1:3,n);

J(:,2*n-1) = R_prefix * dp_dlv_n;
J(:,2*n)   = R_prefix * dp_dalpha_n;
end
