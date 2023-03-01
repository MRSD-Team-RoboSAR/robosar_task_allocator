import matplotlib.pyplot as plt

PGART = [5.052631, 5.737158, 7.543252, 3.784267, 4.569892, 2.512768, 4.811833, 4.412153, 8.176324, 2.300674, 3.617383, 10.974359, 3.022508, 4.94801, 7.49543]
RRT = [33.33975, 13.548388, 24.565218, 16.823406, 7.317073, 35.34446, 23.268112, 33.792183, 63.75523, 33.452538, 30.693846, 12.100965, 22.32913, 19.406197, 30.689476]
sot = [PGART, RRT]

boxprops = dict(linewidth=2.5)
meanlineprops = dict(linewidth=2.5)
plt.figure(figsize=(8, 4), dpi=80)
plt.boxplot(sot, widths=0.5)
plt.xticks([1, 2], ['PGART', 'RRT Exploration'])
plt.title("Node Generation Efficiency")
plt.xlabel("Task Generation Method")
plt.ylabel("% nodes pruned")

plt.show()