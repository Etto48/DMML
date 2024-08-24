import colorsys
import numpy as np
import matplotlib.pyplot as plt

def str_to_rgb(color: str) -> tuple[float, float, float]:
    assert color[0] == "#"
    r = int(color[1:3], 16) / 255
    g = int(color[3:5], 16) / 255
    b = int(color[5:7], 16) / 255
    return r, g, b

def rgb_to_str(r: float, g: float, b: float) -> str:
    assert 0 <= r <= 1
    assert 0 <= g <= 1
    assert 0 <= b <= 1
    r = round(r * 255)
    g = round(g * 255)
    b = round(b * 255)
    return f"#{r:02X}{g:02X}{b:02X}"

def avg_colors(color_1: str, color_2: str) -> str:
    c1 = str_to_rgb(color_1)
    c2 = str_to_rgb(color_2)
    avg = [(c1[i] + c2[i]) / 2 for i in range(3)]
    return rgb_to_str(*avg)

def rgb_to_name(r: float, g: float, b: float) -> str:
    colors = [
        ("red", (1, 0, 0)),
        ("green", (0, 1, 0)),
        ("blue", (0, 0, 1)),
        ("yellow", (1, 1, 0)),
        ("cyan", (0, 1, 1)),
        ("magenta", (1, 0, 1)),
    ]
    def distance_hue(h1: float, h2: float) -> float:
        d = abs(h1 - h2)
        return min(d, 1 - d)
    closest = None
    distance_to_closest = float("inf")
    hsv = colorsys.rgb_to_hsv(r, g, b)
    for name, color in colors:
        hsv_color = colorsys.rgb_to_hsv(*color)
        d = distance_hue(hsv[0], hsv_color[0])
        if d < distance_to_closest:
            closest = name
            distance_to_closest = d
    return closest

def random_color() -> str:
    
    min_saturation = 0.3
    min_value = 0.8
    
    h = np.random.random()
    s = np.random.random() * (1 - min_saturation) + min_saturation
    v = np.random.random() * (1 - min_value) + min_value
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return rgb_to_str(r, g, b)

RANDOM_COLORS_N = 1024
RANDOM_COLORS = [random_color() for _ in range(RANDOM_COLORS_N)]

if __name__ == "__main__":
    sqrt_n = int(np.ceil(RANDOM_COLORS_N**0.5))
    colors = np.zeros((sqrt_n**2, 3))
    avg_color = np.zeros(3)
    for i, color in enumerate(RANDOM_COLORS):
        color = str_to_rgb(color)
        colors[i,0] = color[0]
        colors[i,1] = color[1]
        colors[i,2] = color[2]
        avg_color += color
    avg_color /= RANDOM_COLORS_N
    plt.subplot(1, 2, 1)
    plt.imshow(colors.reshape(sqrt_n, sqrt_n, 3))
    plt.title(f"{RANDOM_COLORS_N} random colors")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(np.array([[avg_color]]))
    plt.title(f"Average color {rgb_to_str(*avg_color)}")
    plt.axis("off")
    plt.show()
    
    