# MDP Maze Solving (Autumn 2025)

This project implements a **Value Iteration** algorithm to solve a maze using **Markov Decision Process (MDP)**.  
It computes the **optimal value function** and **optimal policy**, and visualizes both the **policy grid** and **state value heatmap**.  
Additionally, the **optimal path** is plotted on top of both grids.

---

## Dependencies

Make sure you have the following Python libraries installed:

- Python >= 3.8
- numpy
- matplotlib
- os (built-in)

You can install missing libraries via pip:

```bash
pip install numpy matplotlib
```
## Project Structure
```bash
Maze_solve/
│
├── main.py             
├── README.md           
└── outcome/            
```

## How to Run?
1. Open a terminal and navigate to the project folder:
```bash
cd/ path/to/Maze_solve
```
2. Run the main script:
```bash
python maze.py
```
3. Outputs will be saved in the auto-created folder: outcome/

- policy_grid.png → Maze with optimal policy arrows
- value_grid.png → Heatmap of the state value function
- policy_path.png → Optimal path plotted on the policy grid
- value_path.png → Optimal path plotted on the value grid
