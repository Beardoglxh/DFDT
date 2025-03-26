import pandas as pd
import matplotlib.pyplot as plt

# 使用新的数据字典
data = {
    'max_action_num': [5, 10, 15, 20, 25],
    'Halfcheetah_medium': [71.084, 72.89, 71.107, 73.747, 73.12],
    'Halfcheetah_replay': [57.508, 64.734, 63.063, 63.69, 68.282],
    'Halfcheetah_expert': [102.068, 100.943, 90.068, 93.972, 95.464],
    'Hopper_medium': [94.915, 100.04, 99.214, 99.778, 99.646],
    'Hopper_replay': [101.118, 99.944, 90.904, 97.29, 101.332],
    'Hopper_expert': [113.593, 113.52, 113.472, 113.532, 112.691],
    'Walker2d_medium': [22.642, 50.825, 67.323, 62.703, 65.780],
    'Walker2d_replay': [24.220, 81.204, 16.225, 12.987, 8.570],
    'Walker2d_expert': [34.820, 25.344, 110.225, 112.176, 50.389]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 创建图形和子图，设置为横向排列
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 函数用于突出显示最高点
def highlight_max(ax, x, y, label):
    max_index = y.idxmax()
    ax.plot(x[max_index], y[max_index], marker='o', markersize=10, color='red', label='_nolegend_')

# 绘制 Halfcheetah 的图表
axs[0].plot(df['max_action_num'], df['Halfcheetah_medium'], marker='o', label='Medium', linestyle='-')
highlight_max(axs[0], df['max_action_num'], df['Halfcheetah_medium'], 'Medium')
axs[0].plot(df['max_action_num'], df['Halfcheetah_replay'], marker='x', label='Replay', linestyle='--')
highlight_max(axs[0], df['max_action_num'], df['Halfcheetah_replay'], 'Replay')
axs[0].plot(df['max_action_num'], df['Halfcheetah_expert'], marker='^', label='Expert', linestyle=':')
highlight_max(axs[0], df['max_action_num'], df['Halfcheetah_expert'], 'Expert')
axs[0].set_title('Halfcheetah Performance')
axs[0].set_xlabel('Max Action Number')
axs[0].set_ylabel('Performance')
axs[0].legend()
axs[0].grid()

# 绘制 Hopper 的图表
axs[1].plot(df['max_action_num'], df['Hopper_medium'], marker='o', label='Medium', linestyle='-')
highlight_max(axs[1], df['max_action_num'], df['Hopper_medium'], 'Medium')
axs[1].plot(df['max_action_num'], df['Hopper_replay'], marker='x', label='Replay', linestyle='--')
highlight_max(axs[1], df['max_action_num'], df['Hopper_replay'], 'Replay')
axs[1].plot(df['max_action_num'], df['Hopper_expert'], marker='^', label='Expert', linestyle=':')
highlight_max(axs[1], df['max_action_num'], df['Hopper_expert'], 'Expert')
axs[1].set_title('Hopper Performance')
axs[1].set_xlabel('Max Action Number')
axs[1].set_ylabel('Performance')
axs[1].legend()
axs[1].grid()

# 绘制 Walker2d 的图表
axs[2].plot(df['max_action_num'], df['Walker2d_medium'], marker='o', label='Medium', linestyle='-')
highlight_max(axs[2], df['max_action_num'], df['Walker2d_medium'], 'Medium')
axs[2].plot(df['max_action_num'], df['Walker2d_replay'], marker='x', label='Replay', linestyle='--')
highlight_max(axs[2], df['max_action_num'], df['Walker2d_replay'], 'Replay')
axs[2].plot(df['max_action_num'], df['Walker2d_expert'], marker='^', label='Expert', linestyle=':')
highlight_max(axs[2], df['max_action_num'], df['Walker2d_expert'], 'Expert')
axs[2].set_title('Walker2d Performance')
axs[2].set_xlabel('Max Action Number')
axs[2].set_ylabel('Performance')
axs[2].legend()
axs[2].grid()

# 调整布局
plt.tight_layout()
plt.show()
