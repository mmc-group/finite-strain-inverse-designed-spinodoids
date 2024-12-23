clear; clc; 
list=[]
%% Design parameters
for i = 15:1:70
% Fixed parameters
resolution = 100;
waveNumber = 5;
relativeDensity = 0.5;
radialSmearing = 0.3;
angularSmearing = 0.175;
% Variable parameters
thetas = [0,0,i];

% Create grid with defined resolution
discretization = linspace(-resolution/2,resolution/2-1, resolution);
[X,Y,Z] = ndgrid(discretization,discretization,discretization);

% Distances to each point in the grid
R = sqrt(X.^2 + Y.^2 + Z.^2);

%% Cones along X
% Compute angles at each point w.r.t X-axis
theta_X = min(acosd(X./R),acosd(-X./R));
filter_X = smooth(-(theta_X - thetas(1)) * angularSmearing) .* exp(-0.5 * ((R - waveNumber) / radialSmearing).^2);
filter_X(resolution/2+1, resolution/2+1, resolution/2+1) = 0;
% Don't apply smoothing if theta(1) is 0
if thetas(1) == 0
    filter_X = zeros(resolution,resolution,resolution);
end

%% Cones along Y
% Compute angles at each point w.r.t X-axis
theta_Y = min(acosd(Y./R),acosd(-Y./R));
filter_Y = smooth(-(theta_Y - thetas(2)) * angularSmearing) .* exp(-0.5 * ((R - waveNumber) / radialSmearing).^2);
filter_Y(resolution/2+1, resolution/2+1, resolution/2+1) = 0;
% Don't apply smoothing if theta(2) is 0
if thetas(2) == 0
    filter_Y = zeros(resolution,resolution,resolution);
end

%% Cones along Z
% Compute angles at each point w.r.t X-axis
theta_Z = min(acosd(Z./R),acosd(-Z./R));
filter_Z = smooth(-(theta_Z - thetas(3)) * angularSmearing) .* exp(-0.5 * ((R - waveNumber) / radialSmearing).^2);
filter_Z(resolution/2+1, resolution/2+1, resolution/2+1) = 0;
% Don't apply smoothing if theta(3) is 0
if thetas(3) == 0
    filter_Z = zeros(resolution,resolution,resolution);
end

sdf = filter_X + filter_Y + filter_Z;
list=[list sum(sum(sum(sdf)))];
end
plot(linspace(15,70,70-14),list)
function [v] = smooth(a)
v=(1+tanh(a))/2;
end