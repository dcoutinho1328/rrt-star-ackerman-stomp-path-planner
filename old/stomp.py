import math


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def smooth_stomp(path, weight_data=0.1, weight_smooth=0.1, tolerance=0.00001):
    # Copiar o caminho original para um novo caminho de saída
    new_path = [p for p in path]

    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):
            for j in range(2):
                old_val = new_path[i][j]
                new_val = (
                    new_path[i][j]
                    + weight_data * (path[i][j] - new_path[i][j])
                    + weight_smooth
                    * (new_path[i - 1][j] + new_path[i + 1][j] - (2.0 * new_path[i][j]))
                )
                new_path[i] = (
                    new_path[i][0] if j == 1 else new_val,
                    new_path[i][1] if j == 0 else new_val,
                    new_path[i][2],
                )
                change += abs(old_val - new_val)

    return new_path
