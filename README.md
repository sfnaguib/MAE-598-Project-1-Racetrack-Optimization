# MAE-598-Project-1-Racetrack-Optimization
This project tasked us with optimizing a generalized dynamic system problem. For this, I wrote a Python code to optimize the path of a Formula 1 car around a circular track. This code optimizes the path itself as it travels at a constant speed around the track. It makes use of learning rates and loss functions. 

Overview

The script is written in Python, leveraging the torch library, which suggests the use of machine learning or deep learning techniques. It also uses matplotlib for visualization and numpy for numerical computations.

Key Components of the Code

Environment Setup: The script begins by importing necessary libraries and setting the device for computation, which adapts to CUDA if available, indicating GPU usage for intensive computations.
Simulation Parameters: Parameters such as speed, wheelbase, time step, number of steps, track radius, and track width are defined. These parameters are crucial for simulating the vehicle's movement on a track.

Assumptions and Constraints

The script seems to consider a fixed speed and wheelbase, which implies a simplified model of vehicle dynamics.
The circular track with a defined radius and width sets the geometric constraints for the vehicle's path optimization. For this system as well, the speed of the vehicle is kept constant, which is not very realistic as to reduce the time it takes to go through a track acceleration and braking of the vehicle are employed. A future version of this code that I am working on will implement this feature. 

Methods Used for Solving

Optimization Technique: The use of the torch library suggests that gradient-based optimization methods might be used, possibly involving a loss function and backpropagation.
Simulation-Based Approach: The defined time step and number of steps indicate a discrete simulation over time, likely iterating the vehicle's position and adjusting the steering angle to optimize the path.

Future Implications in Motorsport Engineering

Path Optimization: This script could be foundational in developing algorithms for autonomous or semi-autonomous racing vehicles, where optimal pathfinding is crucial.
Real-time Decision Making: If extended to real-time applications, this tool can aid in dynamic decision-making during races.
Vehicle Dynamics Study: By adjusting parameters like speed and wheelbase, the tool can be used to study the impact of different vehicle dynamics on racing performance.
AI-Driven Strategies: Integration with AI can lead to more advanced racing strategies, considering factors like tire wear, fuel consumption, and overtaking maneuvers.
Limitations and Further Development
The current script seems to focus on a very simplified track model. Future developments could include more complex track layouts and varying environmental conditions.
Integration with more detailed vehicle dynamics models can improve the realism and applicability of the simulation.
This analysis is based on the initial portion of the code. A more thorough review of the entire script would provide deeper insights into the specific algorithms and techniques used.
