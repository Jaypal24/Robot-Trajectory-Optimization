#  Assignment 4: Prediction and Dashboard  
**Learning-Based Trajectory Planning for a 2-Link Robotic Arm**

---

##  Overview
This project demonstrates how **machine learning** can be used to approximate **optimized joint-space trajectories** for a robotic arm and significantly reduce computation time.  
Optimized trajectories are first generated using numerical methods, then a **neural network** is trained to predict full trajectories directly from start and goal joint configurations.  
An **interactive Streamlit dashboard** is built to visually compare optimized and learned trajectories.

---

##  Objective
- Integrate **optimization** and **learning** into a trajectory planning pipeline  
- Train a neural network to predict joint-space trajectories  
- Build an interactive dashboard for visualization and comparison  
- Analyze trade-offs between **accuracy** and **computation speed**

---

##  System Description
- **Robot:** 2-Link Planar Robotic Arm  
- **Joints:**  
  - `q1` → First revolute joint  
  - `q2` → Second revolute joint  
- **Trajectory Space:** Joint space  
- **Time Discretization:** Fixed number of time steps (`T = 50`)

---

##  Dataset Preparation
Each data sample consists of:
- **Input (4 values):**
  - `q1_start`, `q2_start`  
  - `q1_end`, `q2_end`  
- **Output:**
  - Full joint-space trajectory  
    [q1(t), q2(t)] for t = 1 ... T


### Dataset Details
- Optimized trajectories generated using linear interpolation  
- Total samples: **500**  
- Train/Test split: **80% / 20%**

---

##  Learning Model
- **Model Type:** Multilayer Perceptron (MLP)  
- **Input Dimension:** 4  
- **Output Dimension:** `2 × T`  
- **Loss Function:** Mean Squared Error (MSE)  
- **Goal:** Learn the structure of optimized trajectories

---

##  Evaluation
- **Metric:** Mean Squared Error (MSE)  
- **Results:**
  - Low error for smooth trajectories  
  - Slight deviation near extreme joint configurations  
  - Prediction time is **instantaneous**

---

##  Interactive Dashboard
The dashboard is implemented using **Streamlit**.

### Features
- Sliders for start and goal joint angles  
- Optimized vs predicted joint-angle trajectories  
- End-effector path visualization  
- Live prediction error display  

### Visualizations
- Joint angles vs time  
- End-effector (x, y) path  
- Direct comparison between optimized and learned trajectories  

---

##  Optimization vs Learning Comparison

| Aspect | Optimization | Learning |
|------|-------------|---------|
| Accuracy | High | Slightly lower |
| Computation Time | Slow | Very fast |
| Real-time Use | NO | YES |
| Scalability | Limited | High |

---

