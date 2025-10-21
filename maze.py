import numpy as np
import matplotlib.pyplot as plt
import os

# 构建迷宫环境
def build_maze():
    maze = np.array([
        [1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,1],
        [0,0,1,1,0,1,0,1],
        [1,0,0,1,1,0,0,1],
        [1,1,0,0,1,0,1,1],
        [1,0,1,0,1,0,0,1],
        [1,0,0,0,0,1,0,0],
        [1,1,1,1,1,1,1,1]
    ])  # 1表示墙，0表示可走

    start, goal = (2, 0), (6, 7)
    return maze, start, goal

ACTIONS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
GAMMA = 0.9
REWARD_STEP = -1

# 状态转移函数
def step(maze, state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= maze.shape[0] or nc < 0 or nc >= maze.shape[1] or maze[nr, nc] == 1:
        return state, REWARD_STEP
    return (nr, nc), REWARD_STEP

# Value Iteration 算法
def value_iteration(maze, start, goal, theta=1e-4):
    V = np.zeros_like(maze, dtype=float)
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        for r in range(maze.shape[0]):
            for c in range(maze.shape[1]):
                if maze[r, c] == 1 or (r, c) == goal:
                    continue
                v = V[r, c]
                V[r, c] = max([
                    REWARD_STEP + GAMMA * V[step(maze, (r, c), a)[0]]
                    for a in ACTIONS
                ])
                delta = max(delta, abs(v - V[r, c]))
        if delta < theta:
            print(f"Value Iteration Converged after {iteration} iterations.")
            break
    return V

# 提取最优策略
def extract_policy(V, maze, goal):
    policy = np.full(maze.shape, ' ')
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 1 or (r, c) == goal:
                continue
            best_a = max(ACTIONS.keys(),
                         key=lambda a: REWARD_STEP + GAMMA * V[step(maze, (r, c), a)[0]])
            policy[r, c] = best_a
    return policy

# 可视化
def visualize_and_save(V, policy, maze, start, goal, out_dir='./outcome'):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6,5))
    plt.imshow(maze, cmap='gray_r')
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 1: continue
            plt.text(c, r, policy[r, c], ha='center', va='center')
    plt.text(start[1], start[0], 'S', color='green', fontweight='bold', ha='center')
    plt.text(goal[1], goal[0], 'G', color='red', fontweight='bold', ha='center')
    plt.title("Optimal Policy via Value Iteration")
    plt.savefig(os.path.join(out_dir, 'policy_grid.png'), bbox_inches='tight', dpi=150)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(V, cmap='plasma', origin='upper')
    plt.colorbar(label='State Value')
    plt.title("State Value Function (V*)")
    plt.savefig(os.path.join(out_dir, 'value_grid.png'), bbox_inches='tight', dpi=150)
    plt.close()

# 新增功能：绘制最优路径连线
def get_optimal_path(policy, start, goal, maze, max_steps=200):
    path = [start]
    current = start
    steps = 0
    while current != goal and steps < max_steps:
        r, c = current
        action = policy[r, c]
        if action not in ACTIONS:
            break
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc
        if maze[nr, nc] == 1:
            break
        current = (nr, nc)
        path.append(current)
        steps += 1
    return path

def plot_path_on_grids(V, policy, maze, path, start, goal, out_dir='./outcome'):
    import os
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6,5))
    plt.imshow(maze, cmap='gray_r')
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r,c] == 1: continue
            plt.text(c, r, policy[r,c], ha='center', va='center', fontsize=10)
    y, x = zip(*path)
    plt.plot(x, y, color='red', linewidth=2.5, marker='o', markersize=5)
    plt.text(start[1], start[0], 'S', color='green', fontweight='bold', ha='center')
    plt.text(goal[1], goal[0], 'G', color='red', fontweight='bold', ha='center')
    plt.title("Optimal Path (on Policy Grid)")
    plt.savefig(os.path.join(out_dir, 'policy_path.png'), bbox_inches='tight', dpi=150)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(V, cmap='plasma', origin='upper')
    plt.colorbar(label='State Value')
    y, x = zip(*path)
    plt.plot(x, y, color='white', linewidth=2.5, marker='o', markersize=5)
    plt.text(start[1], start[0], 'S', color='lime', fontweight='bold', ha='center')
    plt.text(goal[1], goal[0], 'G', color='red', fontweight='bold', ha='center')
    plt.title("Optimal Path (on Value Grid)")
    plt.savefig(os.path.join(out_dir, 'value_path.png'), bbox_inches='tight', dpi=150)
    plt.close()

# main
if __name__ == "__main__":
    maze, start, goal = build_maze()
    V = value_iteration(maze, start, goal)
    policy = extract_policy(V, maze, goal)
    visualize_and_save(V, policy, maze, start, goal)
# 新增功能：绘制最优路径连线
    path = get_optimal_path(policy, start, goal, maze)
    plot_path_on_grids(V, policy, maze, path, start, goal)