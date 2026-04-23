function H_i = makeHomogeneousTransform(R_i, p_i)
% makeHomogeneousTransform  Build 4×4 homogeneous transform H_i
%
%   H_i = makeHomogeneousTransform(R_i, p_i)
%
% Inputs:
%   R_i – 3×3 rotation matrix
%   p_i – 3×1 position vector
%
% Output:
%   H_i – 4×4 homogeneous transform [R_i, p_i; 0 0 0 1]

    % sanity check
    assert(all(size(R_i)==[3,3]), 'R_i must be 3×3');
    assert(isvector(p_i) && numel(p_i)==3, 'p_i must be 3×1');

    H_i = [ R_i,    p_i(:); 
            0  0  0,    1  ];
end
