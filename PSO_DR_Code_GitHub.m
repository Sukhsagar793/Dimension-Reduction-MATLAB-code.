% Notes for reviewers:
%   - This script performs the PSO-based DR only once per run. Because PSO
%     is stochastic, a single run may not reach the exact targetMED; you
%     may need to repeat/run the script multiple times (or wrap it in a
%     loop) until the achieved MED is sufficiently close to targetMED.
%   - After projection the 2D points are centered and normalized to unit
%     average power before MED and SEP-related analyses.
%   - The noise variance (sigma^2 = N0/2) is not required by the DR step
%     itself; it is used later when plotting SEP/ASEP. The normalization
%     step ensures a consistent mapping between MED and Es/N0.

clc; clear;

% 3D constellation points for M-HQAM
points3D = M; % For M = 8, 16, 32, 64, 128, 256, 512, 1024 see paper in section II

% Minimum distance target
targetMED = Delta_target; % See in the table II

% PSO configuration
nParticles = 30; 
nVars = 6; % nVariables = 6 because the 3x2 projection matrix P belong to (R^3*2) has 6 independent elements.
nIters = 100;
lb = -2; 
ub = 2; 
w = 0.7; 
c1 = 1.5; 
c2 = 1.5;

% Initialize particles
pos = lb + (ub-lb) * rand(nParticles, nVars);
vel = zeros(nParticles, nVars);
pbest = pos;
pbestVal = arrayfun(@(i) objFun(pos(i,:), points3D, targetMED), 1:nParticles)';

[~, idx] = min(pbestVal);
gbest = pbest(idx, :); 
gbestVal = pbestVal(idx);

% Main PSO loop
for t = 1:nIters
    for i = 1:nParticles
        r1 = rand(1, nVars); 
        r2 = rand(1, nVars);

        vel(i,:) = w*vel(i,:) + c1*r1.*(pbest(i,:) - pos(i,:)) + ...
                               c2*r2.*(gbest - pos(i,:));

        pos(i,:) = max(min(pos(i,:) + vel(i,:), ub), lb);

        val = objFun(pos(i,:), points3D, targetMED);

        if val < pbestVal(i)
            pbest(i,:) = pos(i,:); 
            pbestVal(i) = val;
        end

        if val < gbestVal
            gbest = pos(i,:); 
            gbestVal = val;
        end
    end

    if mod(t,10) == 0
        fprintf('Iteration %d | Best Error: %.6f\n', t, gbestVal);
    end
end

% Build projection matrix from best solution
projMat = reshape(gbest, 3, 2);
points2D = points3D * projMat;

% Compute achieved minimum distance
medAchieved = min(pdist(points2D));

% Plot projected points
figure;
scatter(points2D(:,1), points2D(:,2), 60, 'filled');
xlabel('I'); ylabel('Q');
axis equal; grid on;

% Complex symbol form
sj = points2D(:,1) + 1i*points2D(:,2);

% ---------- Objective Function ----------
function err = objFun(x, points3D, target)
    proj = reshape(x, 3, 2);
    proj2D = points3D * proj;

    % Center and normalize to maintain unit average power
    proj2D = proj2D - mean(proj2D);
    proj2D = proj2D / sqrt(mean(sum(proj2D.^2,2)));

    dists = pdist(proj2D);
    minDist = min(dists);
    err = (minDist - target)^2;
end




