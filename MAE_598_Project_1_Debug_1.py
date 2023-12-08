import torch
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters for simulation
speed = 10.0  # speed in meters per second
wheelbase = 2.5  # wheelbase of the car in meters
dt = 0.1  # time step in seconds
num_steps = 325  # number of steps in the simulation
track_radius = 50  # radius of the track in meters
track_width = 10.0  # width of the track in meters
learning_rate = 0.0008  # learning rate for the optimizer
iterations = 500  # number of optimization iterations

# Initialize the steering angles as a PyTorch tensor with small random values
steering_angles = torch.randn((num_steps,), device=device) * 0.01
steering_angles.requires_grad = True

# Define the optimizer using the Adam algorithm
optimizer = torch.optim.Adam([steering_angles], lr=learning_rate)

# List to store the loss and path length at each iteration for plotting
loss_history = []
max_steering_angles = []

# Simulation loop
for i in range(iterations):
    optimizer.zero_grad()

    # Initialize state variables (x, y, theta)
    x = torch.tensor([2 * track_radius], dtype=torch.float32, device=device)
    y = torch.tensor([0.0], dtype=torch.float32, device=device)
    theta = torch.tensor([np.pi / 2], dtype=torch.float32, device=device)

    path_length = torch.tensor([0.0], dtype=torch.float32, device=device)
    penalty = torch.tensor([0.0], dtype=torch.float32, device=device)

    # Simulate the car's trajectory
    for angle in steering_angles:
        new_theta = theta + torch.tan(angle) * speed / wheelbase * dt
        dx = torch.cos(new_theta) * speed * dt
        dy = torch.sin(new_theta) * speed * dt
        new_x = x + dx
        new_y = y + dy
        path_length = path_length + torch.sqrt(dx**2 + dy**2)

        # Calculate the distance from the center of the turn
        distance_from_center = torch.sqrt((new_x - track_radius)**2 + new_y**2)

        # Add penalties if the car is outside the track boundaries
        penalty += torch.relu(distance_from_center - (track_radius + track_width / 2))
        penalty += torch.relu((track_radius - track_width / 2) - distance_from_center)

        x, y, theta = new_x, new_y, new_theta

    # Compute the total loss
    loss = path_length  + penalty * 7.5
    loss.backward()
    optimizer.step()
    max_steering_angles.append(torch.max(steering_angles).item())

    # Print the loss for monitoring
    print(f'Iteration {i+1}, Loss: {loss.item()}, Max Steering Angle: {steering_angles.max().item()}, Max Grad: {steering_angles.grad.abs().max().item()}')


    # Store the loss and path length for plotting
    loss_history.append(loss.item())


    # Update the steering angles for the next iteration
    steering_angles = steering_angles.detach().clone().requires_grad_(True)

    # Reset the optimizer with the updated steering angles
    optimizer = torch.optim.Adam([steering_angles], lr=learning_rate)


# Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Loss over iterations')
plt.xlabel('Iteration number')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration number')
plt.legend()
plt.grid(True)
plt.show()

# Plot the maximum steering angle history
plt.figure(figsize=(12, 5))
plt.plot(max_steering_angles, label='Maximum Steering Angle over iterations')  # Use max_steering_angles
plt.xlabel('Iteration number')
plt.ylabel('Maximum Steering Angle')
plt.title('Maximum Steering Angle vs. Iteration number')
plt.legend()
plt.grid(True)
plt.show()


# Detach the optimized steering angles for plotting
optimized_angles = steering_angles.detach().cpu().numpy()

# Plot the optimized path
x = 2 * track_radius
y = 0.0
theta = np.pi / 2
x_vals, y_vals = [x], [y]

# Start plotting from the starting point
starting_point = (2 * track_radius, 0.0)

for angle in optimized_angles:
    theta += np.tan(angle) * speed / wheelbase * dt
    x += np.cos(theta) * speed * dt
    y += np.sin(theta) * speed * dt
    x_vals.append(x)
    y_vals.append(y)

# Draw the track limits
inner_bound = plt.Circle((track_radius, 0), track_radius - track_width / 2, color='gray', fill=False)
outer_bound = plt.Circle((track_radius, 0), track_radius + track_width / 2, color='gray', fill=False)

# Plotting the figure
plt.figure()
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_artist(inner_bound)
ax.add_artist(outer_bound)
ax.plot(x_vals, y_vals, label='Optimized Path', color='blue')
ax.plot(*starting_point, 'go', label='Starting Point')
ax.set_xlim([0, track_radius * 2])
ax.set_ylim([-track_radius, track_radius])
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Optimized Path and Track for a 90-Degree Turn')
ax.legend()
ax.axis('equal')
plt.show()