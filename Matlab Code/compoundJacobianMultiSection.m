function J_comp = compoundJacobianMultiSection(Ls, rd, lv, alpha)
% compoundJacobianMultiSection  Compute the compound (3n × 2n) Jacobian for all
%                              intermediate tip velocities of an n-section
%                              continuum robot with fixed base.
%
%   J_comp = compoundJacobianMultiSection(Ls, rd, lv, alpha)
%
% Inputs:
%   Ls         – 1×n vector of undeformed arc-lengths [L1,…,Ln]
%   rd         – scalar disk-radius parameter
%   lv         – 1×n vector of current cable extensions Δℓ_i
%   alpha      – 1×n vector of bending-plane angles α_i
%
% Output:
%   J_comp     – (3n)×(2n) compound Jacobian, such that
%                [dot p_1; ...; dot p_n] = J_comp * [dotℓ; dotα],
%                where dot p_k is the velocity of the k-th section tip,
%                and the k-th 3-row block of J_comp is the 3×(2n) Jacobian
%                for dot p_k (with zeros in columns 2k+1 to 2n).

n = numel(Ls);
J_comp = zeros(3*n, 2*n);

for k = 1:n
    % Compute Jacobian for the k-th tip using the first k sections
    Ls_k = Ls(1:k);
    lv_k = lv(1:k);
    alpha_k = alpha(1:k);
    J_k_partial = genMultiSecJacobianFixedBase(Ls_k, rd, lv_k, alpha_k);  % 3×(2k)
    
    % Place in the k-th block row: columns 1 to 2k get J_k_partial,
    % columns 2k+1 to 2n are already zero
    row_start = 3*(k-1) + 1;
    col_end = 2*k;
    J_comp(row_start : row_start+2, 1 : col_end) = J_k_partial;
end

end