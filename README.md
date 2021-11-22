# MICRO452 Basics of Mobile Robotics Thymio Project

## Team Members: Ekrem Yüksel, Nikolaj Schmid, Sven Becker, Olivier Deforche

## Steps of the project
### 1. Visualization
### 2. Global Navigation
### 3. Local Navigation
### 4. Filtering
### 5. Motion Control

## Inputs and Outputs of the Steps
### 1. Visualization
#### I: Request to map everything (scan, take a picture)
#### O: --list of positions as [x, y] (start, goal, obstacle edges) [x_th, x_goal, edge_1, edge_2, .... edge_n]
####    --Graph: Matrix with "nodes" corresponding to list indices
### 2. Global Navigation
#### I: list of positions [(x_start, y_start), (x_goal, y_goal), (x1, y1),.... (xn, yn)]
##### ->calculate distances for each reachable node
##### ->calculate optimal path
#### O: optimal path: list of nodes [(x1, y1),.... (xn, yn)]
### 3. Local Navigation
#### I: horizontal sensory data
#### O: offset added to path, i.e. wheel_speed to avoid the obstacle
### 4. Filtering
### 5. Motion Control
#### I: Optimal path, state estimation
#### O: Motor Wheel Speed


z