function [SecCords,SecTips, Xaxis, Yaxis, Zaxis, PosFrames] = ...
          genForwardKinMultiSec(Ls, r_d, Ns, del_lvs, alphas)

% --- ensure row-vectors of same length ---
Ls      = Ls(:).';   
Ns      = Ns(:).';
del_lvs = del_lvs(:).';
alphas  = alphas(:).';
nSec    = numel(Ls);
if isscalar(r_d), r_d = repmat(r_d,1,nSec); else r_d = r_d(:).'; end

% --- preallocate outputs ---
SecCords  = [];
Xaxis     = zeros(3, nSec+1);
Yaxis     = zeros(3, nSec+1);
Zaxis     = zeros(3, nSec+1);
PosFrames = zeros(3, nSec+1);

% --- base frame axes/origin ---
Xaxis(:,1)    = [1;0;0];
Yaxis(:,1)    = [0;1;0];
Zaxis(:,1)    = [0;0;1];
PosFrames(:,1)= [0;0;0];

Tbase = eye(4);

for i = 1:nSec
  % 1) compute this section in its own local frame
  [curveLocal, tipLocal, theta] = ...
      forwardKinematicsCC(Ls(i), r_d(i), Ns(i), del_lvs(i), alphas(i));
  
  % 2) lift to homogeneous and map into world
  nPts     = size(curveLocal,1);
  homL     = [curveLocal, ones(nPts,1)]';          % 4×nPts
  homW     = Tbase * homL;                         % 4×nPts
  SecCords = [SecCords; homW(1:3,:)'];             % append N×3

  % 3) build the **local** bend transform
  %    axis in local frame = cross(e1, tipLocal)
  axisL = cross([1;0;0], tipLocal);
  if norm(axisL)<eps, axisL = [0;0;1]; end
  axisL = axisL / norm(axisL);
  Rloc   = rodRotMat(axisL, theta);                % 3×3 %%% some issue
  %noticed in this formulat
  %Rloc   = rotation_matrix_direct_formula(axisL, theta); 
  
  %    translation in local frame = tipLocal
  HsectLocal = [Rloc, tipLocal; 0 0 0 1];
  
  % 4) update cumulative world frame
  Tbase = Tbase * HsectLocal;

  % 5) record new frame’s axes & origin
  Xaxis(:,i+1)    = Tbase(1:3,1);
  Yaxis(:,i+1)    = Tbase(1:3,2);
  Zaxis(:,i+1)    = Tbase(1:3,3);
  PosFrames(:,i+1)= Tbase(1:3,4);
      % Record this section's tip in world frame
    SecTips(:,i) = Tbase(1:3,4);
end
end

