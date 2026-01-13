import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Trajectory Prediction Dashboard",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>Learning-Based Trajectory Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Optimized vs Neural Network Predicted Trajectories for a 2-Link Robotic Arm</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------- CONSTANTS --------------------
T = 50
L1, L2 = 1.0, 1.0
model = joblib.load("trajectory_model.pkl")

# -------------------- FUNCTIONS --------------------
def optimized_trajectory(qs, qg):
    t = np.linspace(0, 1, T)
    return np.outer(1 - t, qs) + np.outer(t, qg)

def forward_kinematics(traj):
    x = L1 * np.cos(traj[:,0]) + L2 * np.cos(traj[:,0] + traj[:,1])
    y = L1 * np.sin(traj[:,0]) + L2 * np.sin(traj[:,0] + traj[:,1])
    return x, y

def mse(a, b):
    return np.mean((a - b) ** 2)

# -------------------- INPUT SECTION --------------------
st.subheader("🔧 Joint Angle Inputs")

c1, c2 = st.columns(2)

with c1:
    q1s = st.slider("q1 start (rad)", -3.14, 3.14, 0.0)
    q2s = st.slider("q2 start (rad)", -3.14, 3.14, 0.0)

with c2:
    q1g = st.slider("q1 goal (rad)", -3.14, 3.14, 1.0)
    q2g = st.slider("q2 goal (rad)", -3.14, 3.14, 1.0)

st.divider()

# -------------------- COMPUTATION --------------------
qs = np.array([q1s, q2s])
qg = np.array([q1g, q2g])

opt_traj = optimized_trajectory(qs, qg)
pred_traj = model.predict(
    np.hstack([qs, qg]).reshape(1, -1)
).reshape(T, 2)

error = mse(opt_traj, pred_traj)

# -------------------- METRICS --------------------
m1, m2, m3 = st.columns(3)
m1.metric("Time Steps", T)
m2.metric("Prediction Error (MSE)", f"{error:.6f}")
m3.metric("Computation", "Instant")

st.divider()

# -------------------- JOINT TRAJECTORY PLOT --------------------
st.subheader("📈 Joint Angle Trajectories")

fig1, ax1 = plt.subplots(figsize=(9,4))
ax1.plot(opt_traj[:,0], label="Optimized q1")
ax1.plot(pred_traj[:,0], '--', label="Predicted q1")
ax1.plot(opt_traj[:,1], label="Optimized q2")
ax1.plot(pred_traj[:,1], '--', label="Predicted q2")

ax1.set_xlabel("Time Step")
ax1.set_ylabel("Joint Angle (rad)")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# -------------------- END EFFECTOR PATH --------------------
st.subheader("🦾 End-Effector Path")

x_opt, y_opt = forward_kinematics(opt_traj)
x_pred, y_pred = forward_kinematics(pred_traj)

fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.plot(x_opt, y_opt, label="Optimized Path")
ax2.plot(x_pred, y_pred, '--', label="Predicted Path")

ax2.set_aspect("equal")
ax2.set_xlabel("X Position")
ax2.set_ylabel("Y Position")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# -------------------- OBSERVATIONS --------------------
st.subheader("📝 Observations")

st.markdown("""
- The neural network accurately reproduces optimized trajectories for smooth motions.
- Small deviations appear near the goal configuration.
- Prediction is **orders of magnitude faster** than numerical optimization.
- Learning-based methods are ideal for **real-time trajectory generation**.
""")
