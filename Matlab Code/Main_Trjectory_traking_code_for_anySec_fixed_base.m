clc; close all; clear all;
% Example usage of genForwardKinMultiSec

% --- 1) Define your robot’s parameters for each section ---
% Let’s say we have two sections:
Ls      = [0.7, 0.7,0.7,0.7,0.7];         % section lengths [m]
% r_d     = [0.005, 0.005, 0.005];       % disk radii [m] (same for both)
r_d     = 0.005;
Ns      = [15, 12,12,12,12];             % number of disks per section
% del_lvs = [0.012, 0.018,0.01];       % cable extensions [m]
% alphas  = [pi/4, 0,-pi/4];         % bending-plane angles [rad]
%%
%% Forwar Kinematic with fixed base
Tbase = eye(4);
basePos     = [0.0; 0; 0];    % in the *base* frame
baseAngles = [0; 0; 0.00];    % in the *base* frame


numSec = length(Ls);
numPoints = 1000;

% Define different bounds for each section
% Format: each row is [lower_bound, upper_bound] for that section

% Alpha bounds for each section (radians)
alpha_bounds = [0.001,   1*pi;      % Section 1: 0.01 to π
                0.001,   1*pi;    % Section 2: 0.02 to 1.5π
                0.001,   2*pi; 
                0.001,   2*pi;% Section 3: 0.03 to 2π
                0.001,   2*pi];   % Section 4: 0.04 to 2.5π


% LV bounds for each section (length/velocity)
lv_bounds = [0.0001, 0.002;      % Section 1: -0.0001 to -0.005
             0.0002, 0.003;      % Section 2: -0.0002 to -0.010
             0.0003, 0.005;
             0.0001, 0.002;% Section 3: -0.0003 to -0.015
             0.0004, 0.006];     % Section 4: -0.0004 to -0.020

[alphas, del_lvs] = generate_trajectories_inputs(numSec, alpha_bounds, lv_bounds, numPoints);
%%

% --- 2) Run the multi-section FK routine ---
[SecCords,SecTips, Xaxis, Yaxis, Zaxis, PosFrames] = ...
    genForwardKinMultiSec(Ls, r_d, Ns, del_lvs, alphas);
J = compoundJacobianMultiSection(Ls, r_d, del_lvs, alphas);
% Jsymb=JcobianThreeSecfixedbaseSymb(Ls,r_d,del_lvs,alphas);
% SecCords   is M×3, all curve points concatenated
% Xaxis/Yaxis/Zaxis are 3×(nSec+1) frame axes
% PosFrames  is 3×(nSec+1) frame origins
%%
% --- 3) Plot the backbone curve ---
figure; hold on; grid minor;box on; axis equal
plot3( SecCords(:,1), SecCords(:,2), SecCords(:,3),'o','MarkerSize',6, 'LineWidth',1.5 );
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Multi‐Section Continuum Robot Backbone');

% --- 4) Overlay each section’s local frame axes ---
nSec = size(Xaxis,2)-1;
scale = 0.2;  % axis length for visualization
for i = 1:(nSec+1)
    O = PosFrames(:,i);
    % draw X, Y, Z axes at each frame origin
    quiver3(O(1),O(2),O(3), ...
        scale*Xaxis(1,i), scale*Xaxis(2,i), scale*Xaxis(3,i), ...
        'r','LineWidth',1.2,'MaxHeadSize',0.5);
    quiver3(O(1),O(2),O(3), ...
        scale*Yaxis(1,i), scale*Yaxis(2,i), scale*Yaxis(3,i), ...
        'g','LineWidth',1.2,'MaxHeadSize',0.5);
    quiver3(O(1),O(2),O(3), ...
        scale*Zaxis(1,i), scale*Zaxis(2,i), scale*Zaxis(3,i), ...
        'k','LineWidth',1.2,'MaxHeadSize',0.5);
end

%legend('Backbone','X‐axis','Y‐axis','Z‐axis','Location','bestoutside');
view(3)
%%
% Initialize output arrays
numPoints = size(alphas, 1);
%numSec = size(alpha_traj, 2);

% Preallocate cell arrays to store results for each trajectory point
SecCords_all = cell(numPoints, 1);
SecTips_all = zeros(numPoints, size(SecTips,1), size(SecTips,2));  % Adjust dimensions
Xaxis_all = cell(numPoints, 1);
Yaxis_all = cell(numPoints, 1);
Zaxis_all = cell(numPoints, 1);
PosFrames_all = cell(numPoints, 1);

% Loop through each trajectory point
for i = 1:numPoints
    % Extract current trajectory values for all sections
    current_lv_traj = del_lvs(i, :);      % 1 x numSec
    current_alpha_traj = alphas(i, :); % 1 x numSec
    
    % Call the forward kinematics function
    [SecCords, SecTips, Xaxis, Yaxis, Zaxis, PosFrames] = ...
        genForwardKinMultiSec(Ls, r_d, Ns, current_lv_traj, current_alpha_traj);
    
    % Store results
%     SecCords_all{i} = SecCords;
    SecTips_all(i, :, :) = SecTips;  % Store as 3D array
%     Xaxis_all{i} = Xaxis;
%     Yaxis_all{i} = Yaxis;
%     Zaxis_all{i} = Zaxis;
%     PosFrames_all{i} = PosFrames;
    
    % Optional: Display progress
    if mod(i, 100) == 0
        fprintf('Processed %d/%d trajectory points\n', i, numPoints);
    end
end
%% ploting trajrctory
% Create 3D plot with all sets
figure('Position', [100, 100, 1000, 800]);
hold on;
grid on;

% Colors for each set
colors = lines(5);
set_labels = {'Set 1', 'Set 2', 'Set 3', 'Set 4','Set 5'};

% Plot each set in 3D
for set = 1:numSec
    x = squeeze(SecTips_all(:, 1, set));
    y = squeeze(SecTips_all(:, 2, set));
    z = squeeze(SecTips_all(:, 3, set));
    
    plot3(x, y, z, 'Color', colors(set, :), 'LineWidth', 2, ...
          'DisplayName', set_labels{set});
    
    % Plot start and end points
    scatter3(x(1), y(1), z(1), 100, colors(set, :), 'filled', ...
             'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    scatter3(x(end), y(end), z(end), 100, colors(set, :), 's', 'filled', ...
             'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end

% LaTeX formatting
xlabel('$X$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$Y$', 'Interpreter', 'latex', 'FontSize', 14);
zlabel('$Z$', 'Interpreter', 'latex', 'FontSize', 14);
title('3D Trajectories of All Coordinate Sets', 'Interpreter', 'latex', 'FontSize', 16);

legend('Interpreter', 'latex', 'FontSize', 12, 'Location', 'best');
axis equal;

% Set nice 3D view
view(3);
rotate3d on;
hold off;
%% velocity
% Parameters
dt = 1/1000;  % Time step
num_samples = 1000;
num_sets = numSec;

% Initialize velocity array (999x3x4 since differentiation reduces by 1)
velocity_all = zeros(num_samples-1, 3, num_sets);

% Calculate velocity for each set
for set = 1:num_sets
    for dim = 1:3
        % Differentiate position to get velocity
        position_data = squeeze(SecTips_all(:, dim, set));
        velocity_all(:, dim, set) = diff(position_data) / dt;
    end
end

% Display statistics
fprintf('Velocity calculation completed:\n');
fprintf('Velocity array size: %dx%dx%d\n', size(velocity_all));
for set = 1:num_sets
    fprintf('Set %d - Velocity magnitude range: [%.4f, %.4f] units/s\n', ...
            set, min(vecnorm(squeeze(velocity_all(:,:,set)), 2, 2)), ...
            max(vecnorm(squeeze(velocity_all(:,:,set)), 2, 2)));
end

%% closed loop control
lv_initial=del_lvs(1,:)';
alphas_initial=alphas(1,:)';
v=zeros(2*numSec,1);
v(1:2:end)=lv_initial;
v(2:2:end)=alphas_initial;
%v=[lv_initial;alphas_initial]; % initial conditions 
x_tracking=[];
TdcrSec=[];
Time=[0];
timeTemp=0;
%% Modified Control Loop with Frame Data Storage
% Preallocate storage for frame data across all iterations
numTimeSteps = size(SecTips_all, 1) - 1;
TdcrSec = [];
PosFrames_all = cell(1, numTimeSteps);
Xaxis_all = cell(1, numTimeSteps);
Yaxis_all = cell(1, numTimeSteps);
Zaxis_all = cell(1, numTimeSteps);

lv_initial = del_lvs(1, :)';
alphas_initial = alphas(1, :)';
v = zeros(2 * numSec, 1);
v(1:2:end) = lv_initial;
v(2:2:end) = alphas_initial;
x_tracking = [];
Time = [0];
timeTemp = 0;

for k = 1:numTimeSteps
    J = compoundJacobianMultiSection(Ls, r_d, v(1:2:end, k), v(2:2:end, k));
    [TdcrSecCurrent, SecTipsfeedback, Xaxis, Yaxis, Zaxis, PosFrames] = ...
        genForwardKinMultiSec(Ls, r_d, Ns, v(1:2:end, k), v(2:2:end, k));
    x = reshape(SecTipsfeedback, 3 * numSec, 1);
    xd = cell2mat(arrayfun(@(i) squeeze(SecTips_all(k, :, i))', 1:numSec, 'UniformOutput', false)');
    velocity = cell2mat(arrayfun(@(i) squeeze(velocity_all(k, :, i))', 1:numSec, 'UniformOutput', false)');
    err = xd - x;
    x_tracking = [x_tracking, x];
    TdcrSec = [TdcrSec, TdcrSecCurrent];
    
    % Store frame data for this iteration (using exact index k)
    PosFrames_all{k} = PosFrames;
    Xaxis_all{k} = Xaxis;
    Yaxis_all{k} = Yaxis;
    Zaxis_all{k} = Zaxis;
    
    v(:, k + 1) = v(:, k) + (1) * pinv(J) * (velocity + 30 * err) * dt;
    timeTemp = timeTemp + dt;
    Time = [Time, timeTemp];
end
%% General 3D Plotting of Reference and Tracked Trajectories
% This code plots reference trajectories from SecTips_all and tracked
% trajectories from x_tracking on the same 3D figure. It is general for
% any numSec (number of sections) without modifications.
%
% Assumptions:
% - SecTips_all: numPoints x 3 x numSec (reference tips: rows=time, cols=XYZ, pages=sections)
% - x_tracking: (3*numSec) x (numPoints-1) (tracked tips: rows=XYZ per section stacked, cols=time steps)
% - numSec derived automatically from size(SecTips_all,3)
%
% Usage: Run this after the simulation loop.

% Extract dimensions
numPoints = size(SecTips_all, 1);
numSec = size(SecTips_all, 3);
numTrackPoints = size(x_tracking, 2);  % Should be numPoints-1

% Create 3D plot
figure('Position', [100, 100, 1200, 900]);
hold on;
grid on;
axis equal;

% Colors for sections (repeat if numSec > 7)
colors = lines(max(7, numSec));
lineStyles = {'-', '--'};  % Solid for reference, dashed for tracked

% Labels for sections
set_labels = cell(numSec, 1);
for set = 1:numSec
    set_labels{set} = sprintf('Section %d', set);
end

% Plot reference trajectories (full numPoints)
for set = 1:numSec
    % Reference: plot full trajectory
    x_ref = squeeze(SecTips_all(:, 1, set));
    y_ref = squeeze(SecTips_all(:, 2, set));
    z_ref = squeeze(SecTips_all(:, 3, set));
    
    plot3(x_ref, y_ref, z_ref, 'Color', colors(set, :), ...
          'LineStyle', lineStyles{1}, 'LineWidth', 2, ...
          'DisplayName', sprintf('%s Ref', set_labels{set}));
    
    % Mark start and end points for reference
    scatter3(x_ref(1), y_ref(1), z_ref(1), 100, colors(set, :), ...
             'filled', 'Marker', 'o', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
             'HandleVisibility', 'off');
    scatter3(x_ref(end), y_ref(end), z_ref(end), 100, colors(set, :), ...
             'filled', 'Marker', 's', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
             'HandleVisibility', 'off');
end

% Plot tracked trajectories (numPoints-1 points, aligned to first numTrackPoints of ref)
for set = 1:numSec
    % Tracked: extract from x_tracking
    row_start = 3*(set-1) + 1;
    row_end = row_start + 2;
    x_track = x_tracking(row_start, :);
    y_track = x_tracking(row_start+1, :);
    z_track = x_tracking(row_start+2, :);
    
    plot3(x_track, y_track, z_track, 'Color', colors(set, :), ...
          'LineStyle', lineStyles{2}, 'LineWidth', 4, ...
          'DisplayName', sprintf('%s Track', set_labels{set}));
    
    % Mark start and end points for tracked
    scatter3(x_track(1), y_track(1), z_track(1), 100, colors(set, :), ...
             '^', 'MarkerFaceColor', colors(set, :), 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1.5, 'HandleVisibility', 'off');
    scatter3(x_track(end), y_track(end), z_track(end), 100, colors(set, :), ...
             'd', 'MarkerFaceColor', colors(set, :), 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1.5, 'HandleVisibility', 'off');
end

% LaTeX formatting for axes and title
xlabel('$X$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$Y$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
zlabel('$Z$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
title('3D Reference and Tracked Trajectories for All Sections', ...
      'Interpreter', 'latex', 'FontSize', 16);

% Legend
legend('Interpreter', 'latex', 'FontSize', 11, 'Location', 'bestoutside');

% Set 3D view and interactivity
view(45, 30);
rotate3d on;
hold off;

% Optional: Print summary
%fprintf('Plotted %d sections: Reference (solid lines, circles/squares) vs. Tracked (dashed lines, triangles/diamonds).\n', numSec);
%% General Time-Series Plotting of Reference and Tracked Coordinates
% This code creates subplots for each section's X, Y, Z coordinates vs. time,
% plotting both reference (from SecTips_all) and tracked (from x_tracking)
% trajectories on the same axes. Layout: numSec rows x 3 columns (X,Y,Z).
% Automatically adjusts for any numSec without modifications.
%
% Assumptions:
% - SecTips_all: numPoints x 3 x numSec (reference tips)
% - x_tracking: (3*numSec) x (numPoints-1) (tracked tips)
% - Time: vector of length numPoints (full time for reference and tracking)
%
% Usage: Run this after the simulation loop. Assumes variables from main script.

% Extract dimensions
numPoints = size(SecTips_all, 1);
numSec = size(SecTips_all, 3);
numTrackPoints = size(x_tracking, 2);  % numPoints-1

% Time vectors from existing Time array
time_ref = Time(1:numPoints);      % Full for reference
time_track = Time(1:numTrackPoints);  % For tracking (first numTrackPoints)

% Generate yLabels dynamically for any numSec
yLabels = cell(3*numSec, 1);
for set = 1:numSec
    for coord = 1:3
        idx = (set-1)*3 + coord;
        yLabels{idx} = sprintf('\\zeta_{%d}(%d)', set, coord);
    end
end

% Font-size parameters (tweak these)
labelFontSize  = 14;   % axis labels
tickFontSize   = 14;   % tick labels
legendFontSize = 12;   % legend text

% Create figure with subplots: numSec rows x 3 columns
fig = figure('Position', [100, 100, 1200, 800 * numSec / 4]);  % Scale height with numSec

% Preallocate axes handles
ax = gobjects(numSec, 3);

for set = 1:numSec
    for coord = 1:3
        subplot_idx = (set-1)*3 + coord;
        ax(set, coord) = subplot(numSec, 3, subplot_idx);
        
        hold on;
        grid minor;
        
        % Reference trajectory (Desired)
        ref_data = squeeze(SecTips_all(:, coord, set));
        plot(time_ref, ref_data, 'Color', [0, 0.4470, 0.7410], ...
             'LineStyle', '-', 'LineWidth', 2, ...
             'DisplayName', 'Desired Trajectory');
        
        % Tracked trajectory (Actual)
        row_start = 3*(set-1) + coord;
        track_data = x_tracking(row_start, :);
        plot(time_track, track_data, 'Color', [0.8500, 0.3250, 0.0980], ...
             'LineStyle', '--', 'LineWidth', 2, ...
             'DisplayName', 'Actual Trajectory');
        
        % Set x-limits
        xlim([min(Time) max(Time)]);
        
        % Axis labels
        hx = xlabel('Time (s)', 'Interpreter', 'latex');
        hy = ylabel(['$' yLabels{subplot_idx} '$(mm)'], 'Interpreter', 'latex');
%         set(hx, 'FontSize', labelFontSize);
%         set(hy, 'FontSize', labelFontSize);
%         
%         % Tick labels
%         set(gca, 'FontSize', tickFontSize);
        
        hold off;
    end
end

% Shared legend (positioned on the first subplot, normalized figure units)
legend(ax(1,1), {'Desired Trajectory', 'Actual Trajectory'}, ...
       'Orientation', 'horizontal', ...
       'Interpreter', 'latex', ...
       'FontSize', legendFontSize, ...
       'Units', 'normalized', ...
       'Position', [0.3679 0.9521 0.2964 0.0400]);

% % Global title if desired
% sgtitle('Reference vs. Tracked Coordinate Trajectories Over Time', ...
%         'FontSize', 16, 'Interpreter', 'latex');

% Optional: Adjust layout for better spacing
%set(gcf, 'Renderer', 'painters');  % For better rendering

%% 3D Segmented Plotting of Continuum Robot Trajectory by Sections with Frames
% Plots the 3D trajectory from TdcrSec (N x 3 matrix: [x, y, z]) with color
% changes every 'points_per_sec' points to represent sections. Automatically
% computes numSec if not provided, but assumes TdcrSec rows are evenly divisible.
% General for any numSec; adjust numSec manually if needed. Now includes frame
% plotting at base and each section tip using stored data at exact selected indices.
%
% Assumptions:
% - TdcrSec: total_points x (3 * numTimeSteps) matrix (rows: points, cols: concatenated x,y,z over time)
% - numSec: number of sections (e.g., 4); derived from length(Ls)
% - Frame data: stored in *_all cell arrays (each {k} is 3 x (numSec+1) for axes/positions at time k)
%
% Usage: Run this after the modified control loop.

% Define or extract numSec (e.g., from Ls)
numSec = length(Ls);

ntdcrPlot = 4;
index_temp = 1:3:size(TdcrSec, 2);
n_temp = length(index_temp);
indices = round(linspace(1, n_temp, ntdcrPlot));
selected_tdcr = index_temp(indices);
% Create figure
figure('Position', [100, 100, 1000, 800]);
hold on;
grid on;
axis equal;
for i = 1:ntdcrPlot
    % Selected time index (exact correspondence to iteration k)
    time_k = indices(i);
    
    TdcrSecplot = TdcrSec(:, selected_tdcr(i):selected_tdcr(i) + 2);  
    
    % If numSec not predefined, you can compute it (assumes even division)
    N = size(TdcrSecplot, 1);
    points_per_sec = N / numSec;  % Assumes exact division (e.g., 604/4=151)

    % Verify even division
    if mod(N, numSec) ~= 0
        warning('TdcrSec rows (%d) not evenly divisible by numSec (%d). Adjusting last segment.', N, numSec);
        points_per_sec = floor(N / numSec);
    end

    % Retrieve frame data for this exact time index
    Xaxis = Xaxis_all{time_k};
    Yaxis = Yaxis_all{time_k};
    Zaxis = Zaxis_all{time_k};
    PosFrames = PosFrames_all{time_k};



    % Generate distinct colors for each section
    colors = lines(numSec);

    % Optional: Plot reference and tracked tips (full trajectories; kept as in original)
    plot3(squeeze(SecTips_all(:, 1, numSec)), squeeze(SecTips_all(:, 2, numSec)), squeeze(SecTips_all(:, 3, numSec)), ...
          'Color', [0, 0.4470, 0.7410], ...
          'LineStyle', '-', 'LineWidth', 2);
    hold on;
    plot3(x_tracking(end - 2, :), x_tracking(end - 1, :), x_tracking(end, :), ...
          'Color', [0.8500, 0.3250, 0.0980], ...
          'LineStyle', '--', 'LineWidth', 3);

    % Plot each section segment with unique color (backbone points at selected time)
    for sec = 1:numSec
        idx_start = (sec - 1) * points_per_sec + 1;
        idx_end = min(sec * points_per_sec, N);  % Handle if not exact
        seg_data = TdcrSecplot(idx_start:idx_end, :);
        
        plot3(seg_data(:, 1), seg_data(:, 2), seg_data(:, 3), ...
              'o', 'Color', colors(sec, :), ...
              'MarkerSize', 6, 'LineWidth', 1.5);
    end

    % Frame plotting: at fixed point (base, frame_idx=1) and each section tip (frame_idx=2 to numSec+1)
    nSec_frames = size(Xaxis, 2) - 1;  % Should equal numSec
    scale = 0.2;  % Axis length for visualization
    
    for frame_idx = 1:(nSec_frames + 1)
        O = PosFrames(:, frame_idx);
        
        % Use consistent colors for all axes across all frames: X red, Y green, Z black
        x_color = 'r';
        y_color = 'g';
        z_color = 'b';
        
        % Draw X, Y, Z axes at each frame origin with fixed axis-specific colors
        quiver3(O(1), O(2), O(3), ...
                scale * Xaxis(1, frame_idx), scale * Xaxis(2, frame_idx), scale * Xaxis(3, frame_idx), ...
                x_color, 'LineWidth', 1.2, 'MaxHeadSize', 0.5);
        quiver3(O(1), O(2), O(3), ...
                scale * Yaxis(1, frame_idx), scale * Yaxis(2, frame_idx), scale * Yaxis(3, frame_idx), ...
                y_color, 'LineWidth', 1.2, 'MaxHeadSize', 0.5);
        quiver3(O(1), O(2), O(3), ...
                scale * Zaxis(1, frame_idx), scale * Zaxis(2, frame_idx), scale * Zaxis(3, frame_idx), ...
                z_color, 'LineWidth', 1.2, 'MaxHeadSize', 0.5);
    end

    % Formatting
    xlabel('$\zeta_1$ (m)', 'FontSize', 14, 'Interpreter', 'latex');
    ylabel('$\zeta_2$ (m)', 'FontSize', 14, 'Interpreter', 'latex');
    zlabel('$\zeta_3$ (m)', 'FontSize', 14, 'Interpreter', 'latex');
    %title('3D Trajectory of Continuum Robot Sections', 'FontSize', 16, 'Interpreter', 'latex');
    %legend('Location', 'best', 'FontSize', 12, 'Interpreter', 'latex');

    % Set 3D view
    view(45, 30);
 rotate3d on;
%     hold off;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
