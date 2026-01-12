import numpy as np

T = 50  # time steps
NUM_SAMPLES = 500

def generate_optimized_trajectory(q_start, q_goal, T=50):
    t = np.linspace(0, 1, T)
    traj = np.outer(1 - t, q_start) + np.outer(t, q_goal)
    return traj  # shape (T, 2)

X, Y = [], []

for _ in range(NUM_SAMPLES):
    q_start = np.random.uniform(-np.pi, np.pi, 2)
    q_goal = np.random.uniform(-np.pi, np.pi, 2)
    
    traj = generate_optimized_trajectory(q_start, q_goal, T)
    
    X.append(np.hstack([q_start, q_goal]))   # input: 4 values
    Y.append(traj.flatten())                  # output: 2*T values

X = np.array(X)
Y = np.array(Y)

np.save("X.npy", X)
np.save("Y.npy", Y)
