import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
file_path = 'results.csv'
df = pd.read_csv(file_path)

# Display the DataFrame to verify data loading
print(df)

# Algorithms and their colors for plotting
algorithms = ["DFS-G", "UCS", "A*", "W-A* (0.3)", "W-A* (0.7)", "W-A* (0.9)"]
colors = ["r", "g", "b", "c", "m", "y"]

# # Create individual scatter plots for each algorithm
# for algo, color in zip(algorithms, colors):
#     plt.figure(figsize=(10, 6))
#     x = df[f"{algo} num of expanded nodes"]
#     y = df[f"{algo} cost"]
#     plt.scatter(x, y, label=algo, color=color)
#     for i, txt in enumerate(df["map"]):
#         plt.annotate(txt, (x[i], y[i]), fontsize=8, color=color)
#     plt.xlabel("Number of Expanded Nodes")
#     plt.ylabel("Cost")
#     plt.title(f"Scatter Plot for {algo}")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# Combined scatter plot with all points
# plt.figure(figsize=(14, 10))

# for algo, color in zip(algorithms, colors):
#     x = df[f"{algo} num of expanded nodes"]
#     y = df[f"{algo} cost"]
#     plt.scatter(x, y, label=algo, color=color)
#     for i in range(len(df)):
#         plt.annotate(df["map"][i], (df[f"{algo} num of expanded nodes"][i], df[f"{algo} cost"][i]), fontsize=8)

# plt.xlabel("Number of Expanded Nodes")
# plt.ylabel("Cost")
# plt.title("Combined Scatter Plot of All Algorithms")
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot all the algorithms for each map
for map_name in df["map"]:
    plt.figure(figsize=(10, 6))
    map_data = df[df["map"] == map_name]
    
    for algo, color in zip(algorithms, colors):
        x = map_data[f"{algo} num of expanded nodes"]
        y = map_data[f"{algo} cost"]
        plt.scatter(x, y, label=f"{algo}", color=color)


    plt.xlabel("Number of Expanded Nodes")
    plt.ylabel("Cost")
    plt.title(f"Scatter Plot for {map_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
