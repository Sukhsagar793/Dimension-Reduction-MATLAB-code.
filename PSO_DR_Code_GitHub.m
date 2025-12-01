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

% 3D constellation points for 16-HQAM
points3D = [ 
     0.0230   -0.0399   -0.0039
     0.0168   -0.0290    0.9178
     0.0167   -0.0290   -0.9236
    -0.4357    0.7547    0.0771
     0.4753   -0.8233   -0.1721
    -0.8712    0.0187   -0.4063
     0.8779   -0.0295    0.3812
    -0.4311   -0.7442    0.4831
     0.4190    0.7652    0.4623
    -0.4294   -0.7852   -0.4356
     0.4297    0.7654   -0.4545
    -0.8896    0.0499    0.5164
     0.8779   -0.0296   -0.5390
     0.4700   -0.8140    0.7538
    -0.4420    0.7656   -0.8425
    -0.4422    0.7659    0.9986]; % For M = 32, 64, 128, 256, 512, 1024 see paper in section II

% Minimum distance target
targetMED = 0.91654; 

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

% Show final points
disp('Final 2D Constellation Points:');
for j = 1:length(sj)
    fprintf('s%d = %.4f + %.4fi;\n', j, real(sj(j)), imag(sj(j)));
end
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
