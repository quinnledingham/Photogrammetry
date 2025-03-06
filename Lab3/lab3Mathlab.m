
close all;
clc;
format long g;
% Input tie points: [xL, yL, xR, yR]
% Given Tie Point Measurements
tie_points = [
    -10.105,  15.011, -103.829,  16.042;
     94.369,  -4.092,   0.868,  -0.022;
    -10.762, -104.711, -100.169, -103.14;
     90.075, -91.378,  -1.607, -87.253;
     -9.489,   96.26, -105.395,  98.706;
     85.42,   103.371,  -9.738, 109.306;
];

% Camera Focal Length (in mm)
c = 153.358 ; % Adjust to your camera's focal length
% Initial Values for Relative Orientation Parameters
b_X = 92;  % Baseline X (known, 92 mm)
b_Y = 0;   % Initial Y baseline
b_Z = 0;   % Initial Z baseline
omega = 0; % Initial rotation angle omega
phi = 0;   % Initial rotation angle phi
kappa = 0; % Initial rotation angle kappa
figure;
subplot(1, 2, 1);
plot(tie_points(:,1), tie_points(:,2), 'ro');
title('Left Image Tie Points');
xlabel('x_L (mm)');
ylabel('y_L (mm)');
grid on;

subplot(1, 2, 2);
plot(tie_points(:,3), tie_points(:,4), 'bo');
title('Right Image Tie Points');
xlabel('x_R (mm)');
ylabel('y_R (mm)');
grid on;
syms xL yL xR yR bY bZ omega phi kappa real

% Define image vectors
vL = [xL; yL; -c];
vR = [xR; yR; -c];

% Rotation matrix (approximate, first-order rotation)
R = [1, kappa, -phi;
    -kappa, 1, omega;
     phi, -omega, 1];

% Relative position vector
B = [b_X; bY; bZ];

% Coplanarity condition
coplanarity = B.' * cross(vL, R * vR);

% Partial derivatives
dF_dby = diff(coplanarity, bY);
dF_dbz = diff(coplanarity, bZ);
dF_domega = diff(coplanarity, omega);
dF_dphi = diff(coplanarity, phi);
dF_dkappa = diff(coplanarity, kappa);

% Display partial derivatives
disp('Partial derivatives:');
disp('dF/dbY ='), pretty(dF_dby)
disp('dF/dbZ ='), pretty(dF_dbz)
disp('dF/dOmega ='), pretty(dF_domega)
disp('dF/dPhi ='), pretty(dF_dphi)
disp('dF/dKappa ='), pretty(dF_dkappa)
syms xL yL xR yR bY bZ omega phi kappa real

% Define image vectors
vL = [xL; yL; -c];
vR = [xR; yR; -c];

% Rotation matrix (approximate, first-order rotation)
R = [1, kappa, -phi;
    -kappa, 1, omega;
     phi, -omega, 1];

% Relative position vector
B = [b_X; bY; bZ];

% Coplanarity condition
coplanarity = B.' * cross(vL, R * vR);

% Partial derivatives
dF_dby = diff(coplanarity, bY);
dF_dbz = diff(coplanarity, bZ);
dF_domega = diff(coplanarity, omega);
dF_dphi = diff(coplanarity, phi);
dF_dkappa = diff(coplanarity, kappa);
% Pixel Spacing (from previous labs)
pixel_spacing = 0.01; % Example value in mm/pixel
% Display the updated table
disp('Y-Parallax in Pixels:');
disp(y_parallax_table);
figure;
bar(tie_point_ids, Y_parallax_pixels);
title('Y-Parallax per Tie Point');
xlabel('Tie Point ID');
ylabel('Y-Parallax (Pixels)');
grid on;
% Calculate the Covariance Matrix
Qxx = inv(A' * A);
% Create Y-Parallax Table
y_parallax_table = table(tie_point_ids, Y_parallax, Y_parallax_pixels, ...
    'VariableNames', {'Tie_Point_ID', 'Y_Parallax_mm', 'Y_Parallax_Pixels'});

% Transform Y-Parallax to Pixel Size
pixel_spacing = 0.01; % Example value in mm/pixel
Y_parallax_pixels = Y_parallax / pixel_spacing;

% Tie point IDs for labeling
tie_point_ids = (1:size(tie_points, 1))';

% Display the table
disp('Y-Parallax Table:');
disp(y_parallax_table);

% Add to the table
y_parallax_table.Y_Parallax_Pixels = Y_parallax_pixels;


% Convert to Correlation Coefficient Matrix
D = sqrt(diag(Qxx));  % Standard deviations of the parameters
correlation_matrix = Qxx ./ (D * D');

% Parameter names for table
parameter_names = {'b_Y', 'b_Z', 'omega', 'phi', 'kappa'};

% Display the Correlation Matrix with Titles
disp('Correlation Coefficient Matrix:');
disp(array2table(correlation_matrix, 'VariableNames', parameter_names, 'RowNames', parameter_names));

% Display partial derivatives
disp('Partial derivatives:');
disp('dF/dbY ='), pretty(dF_dby)
disp('dF/dbZ ='), pretty(dF_dbz)
disp('dF/dOmega ='), pretty(dF_domega)
disp('dF/dPhi ='), pretty(dF_dphi)
disp('dF/dKappa ='), pretty(dF_dkappa)

% Calculate the Covariance Matrix
Qxx = inv(A' * A);

% Convert to Correlation Coefficient Matrix
D = sqrt(diag(Qxx));  % Standard deviations of the parameters
correlation_matrix = Qxx ./ (D * D');

% Parameter names for table
parameter_names = {'b_Y', 'b_Z', 'omega', 'phi', 'kappa'};

% Display the Correlation Matrix with Titles
disp('Correlation Coefficient Matrix:');
disp(array2table(correlation_matrix, 'VariableNames', parameter_names, 'RowNames', parameter_names));



