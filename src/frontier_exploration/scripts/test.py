import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood


def map2image(mapData):
    mapData2 = np.copy(mapData)
    unknown_mask = np.where(mapData == -1)
    mapData2[unknown_mask] = 0.3
    return 1 - mapData2


def informationGain(mapData, point, r):
    x = point[0]
    y = point[1]
    x_min = max(int(x - r), 0)
    x_max = min(int(x + r), mapData.shape[0])
    y_min = max(int(y - r), 0)
    y_max = min(int(y + r), mapData.shape[1])
    infoGain = 0
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if mapData[i][j] == -1:
                infoGain += 1
    if infoGain == 0:
        return infoGain
    return infoGain / ((x_max - x_min) * (y_max - y_min))


def floodFillInfoGrain(mapData, point, r):
    x = point[0]
    y = point[1]
    x_min = max(int(x - r), 0)
    x_max = min(int(x + r), mapData.shape[0])
    y_min = max(int(y - r), 0)
    y_max = min(int(y + r), mapData.shape[1])
    img = map2image(mapData)
    area = img[x_min:x_max, y_min:y_max]
    seed = (x - x_min, y - y_min)
    mask = flood(area, seed, tolerance=0.5)

    resize_mask = np.zeros(mapData.shape)
    contains_free = np.array(mask == 1) & np.array(area == 1)
    if np.any(contains_free):
        infoGain = np.array(mask == 1) & np.array(area == 0.7)
        resize_mask[x_min:x_max, y_min:y_max] = infoGain
        return np.sum(infoGain) / ((x_max - x_min) * (y_max - y_min)), resize_mask
    return 0, resize_mask


mapData = np.zeros((10, 10))
mapData[6:, 4] = 1
mapData[:, :4] = -1
mapData[:, 5] = -1

r = 3
point = np.array([5, 5])

ig = informationGain(mapData, point, r)
print("information gain: ", ig)
ig2, resize_mask = floodFillInfoGrain(mapData, point, r)
print("flood fill gain: ", ig2)

mapData2 = map2image(mapData)
fig = plt.figure()
ax = plt.gca()
ax.set_xticks(np.arange(0, mapData.shape[0], 1))
ax.set_yticks(np.arange(0, mapData.shape[1], 1))
ax.grid(color="k", linewidth=1)
plt.imshow(mapData2, cmap="gray", vmin=0, vmax=1)
plt.imshow(resize_mask, cmap="jet", alpha=0.4)
plt.plot(
    point[0] - 0.5,
    point[1] - 0.5,
    marker="o",
    color="red",
    markersize=20,
    label=str(ig),
)
rect = patches.Rectangle(
    [point[0] - r - 0.5, point[1] + r - 0.5],
    2 * r,
    -2 * r,
    edgecolor="green",
    fill=False,
    linewidth=2,
)
ax.add_patch(rect)
plt.show()
