%% System matrices A, B1, B2, and the cost function parameters Q1, Q2, R11, R12, R21, R22
A = [-0.0366, 0.0271, 0.0188, -0.4555;
     0.0482, -1.0100, 0.0024, -4.0208;
     0.1002, 0.2855, -0.7070, 1.3229;
     0, 0, 1.0000, 0];

B1 = [0.4422; 3.0447; -5.52; 0];
B2 = [0.1761; -7.5922; 4.99; 0];

Q1 = diag([3.5, 2, 4, 5]);
Q2 = diag([1.5, 6, 3, 1]);
R11 = 1;
R22 = 2;
R12 = 0.25;
R21 = 0.6;

S1 = B1 * inv(R11) * B1';
S2 = B2 * inv(R22) * B2';

Z1 = B1 * inv(R11) * R21 * inv(R11) * B1'
Z2 = B2 * inv(R22) * R12 * inv(R22) * B2'

x0 = [0; 0; 0; 1];

% Initialization for policy iteration
K1 = are(A, S1, Q1);
K2 = are(A - S1 * K1, S2, Q2 + K1 * S1 * K1);

% stabilizable-detectable.
if all(real(eig(A - S1 * K1)) < 0)
    disp('Matrix A - S1 * K1 is stable.');
else
    disp('Matrix A - S1 * K1 is NOT stable.');
end
if all(eig(Q2 + K1' * Z1 * K1) >= 0)
    disp('The matrix Q2 + K1''* Z1 * K1 is positive semidefinite.');
else
    disp('The matrix Q2 + K1''* Z1 * K1 is NOT positive semidefinite.');
end
if all(real(eig(A - S1 * K1 - S2 * K2)) < 0)
    disp('Matrix A - S1 * K1 - S2 * K2 is stable.');
else
    disp('Matrix A - S1 * K1 - S2 * K2 is NOT stable.');
end

% Initialize storage arrays
num_iterations = 9;
K1_array = cell(num_iterations, 1);
K2_array = cell(num_iterations, 1);
J1_array = zeros(num_iterations, 1);
J2_array = zeros(num_iterations, 1);

% Store initial values
K1_array{1} = K1;
K2_array{1} = K2;
J1_array(1) = 0.5 * trace(K1);
J2_array(1) = 0.5 * trace(K2);

% Policy Iteration Loop
for i = 2:num_iterations
    % Update K matrices
    K1 = lyap((A - S1 * K1_array{i-1} - S2 * K2_array{i-1})', Q1 + K1_array{i-1} * S1 * K1_array{i-1} + K2_array{i-1} * S2 * K2_array{i-1});
    K2 = lyap((A - S1 * K1_array{i-1} - S2 * K2_array{i-1})', Q2 + K2_array{i-1} * S2 * K2_array{i-1} + K1_array{i-1} * S1 * K1_array{i-1});

    % Store K matrices and compute cost functionals J
    K1_array{i} = K1;
    K2_array{i} = K2;
    J1_array(i) = 0.5 * trace(K1);
    J2_array(i) = 0.5 * trace(K2);
end

%% Simulations and Plots
time = 0:0.05:5; 
x_traj_array = cell(num_iterations, 1);
for i = 1:num_iterations
    x_traj = zeros(length(x0), length(time));
    for j = 1:length(time)
        x_traj(:, j) = expm((A - S1 * K1_array{i} - S2 * K2_array{i}) * time(j)) * x0;
    end
    x_traj_array{i} = x_traj;
end

u1_traj_array = cell(num_iterations, 1);
u2_traj_array = cell(num_iterations, 1);
for k = 1:num_iterations
    u1_traj = -inv(R11) * B1' * K1_array{i} * x_traj_array{k, 1};
    u2_traj = -inv(R22) * B2' * K2_array{i} * x_traj_array{k, 1};
    u1_traj_array{k} = u1_traj;
    u2_traj_array{k} = u2_traj;
end


num_colors = num_iterations;
gradient_colormap = jet(num_colors);

% Plot State Trajectories
for i = 1:size(x0, 1)
    figure;
    legend_labels = cell(num_iterations, 1);
    for k = 1:num_iterations
        plot_color = gradient_colormap(k, :);
        plot(time, x_traj_array{k}(i, :), 'LineWidth', 2, 'Color', plot_color);
        hold on;
        legend_labels{k} = ['Iteration ' num2str(k)];
    end
    legend(legend_labels);
    title(['State Trajectories for Coordinate ' num2str(i)]);
    xlabel('Time');
    ylabel(['x_' num2str(i)]);
    saveas(gcf, sprintf('state_trajectory_coord%d.png', i));
end

% Plot Control Trajectories for Player 1
figure;
legend_labels1 = cell(num_iterations, 1);
for k = 1:num_iterations
    plot_color = gradient_colormap(k, :);
    plot(time, u1_traj_array{k}, 'LineWidth', 2, 'Color', plot_color);
    hold on;
    legend_labels1{k} = ['Iteration ' num2str(k)];
end
legend(legend_labels1);
title('Control Trajectories for Player 1');
xlabel('Time');
ylabel('Control Input (Player 1)');
saveas(gcf, 'control_trajectories_player1.png');

% Plot Control Trajectories for Player 2
figure;
legend_labels2 = cell(num_iterations, 1);
for k = 1:num_iterations
    plot_color = gradient_colormap(k, :);
    plot(time, u2_traj_array{k}, 'LineWidth', 2, 'Color', plot_color);
    hold on;
    legend_labels2{k} = ['Iteration ' num2str(k)];
end
legend(legend_labels2);
title('Control Trajectories for Player 2');
xlabel('Time');
ylabel('Control Input (Player 2)');
saveas(gcf, 'control_trajectories_player2.png');


% Plot J1 and J2
figure;
plot(1:num_iterations, J1_array, 'b-o', 'LineWidth', 2);
hold on;
plot(1:num_iterations, J2_array, 'r-o', 'LineWidth', 2);
legend('J1', 'J2');
title('Convergence of Cost Functionals J1 and J2');
xlabel('Iteration');
ylabel('Cost Functional Value');
grid on;
saveas(gcf, 'J1_J2_convergence.png');

% Create a table for convergence
Iterations = (1:num_iterations)';
T = table(Iterations, J1_array, J2_array, 'VariableNames', {'Iteration', 'J1', 'J2'});
disp(T);

