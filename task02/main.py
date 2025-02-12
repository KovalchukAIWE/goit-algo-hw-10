import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Визначення функції
def f(x):
    return x ** 2

# Межі інтегрування
a = 0  # нижня межа
b = 2  # верхня межа

# ----------------------------
# 1. Побудова графіка функції
# ----------------------------
# Створення діапазону значень для побудови графіка
x_plot = np.linspace(-0.5, 2.5, 400)
y_plot = f(x_plot)

fig, ax = plt.subplots()
ax.plot(x_plot, y_plot, 'r', linewidth=2, label=r'$f(x)=x^2$')

# Заповнення області під кривою (тільки між a і b)
x_fill = np.linspace(a, b, 400)
ax.fill_between(x_fill, f(x_fill), color='gray', alpha=0.3, label='Область інтегрування')

# Малювання вертикальних ліній для меж інтегрування
ax.axvline(x=a, color='gray', linestyle='--', label='Межі інтегрування')
ax.axvline(x=b, color='gray', linestyle='--')

ax.set_xlim([x_plot[0], x_plot[-1]])
ax.set_ylim([0, max(y_plot) + 0.1])
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Графік інтегрування: $f(x)=x^2$ від {} до {}'.format(a, b))
ax.legend()
plt.grid()
plt.show()

# -----------------------------------------------------
# 2. Обчислення інтегралу методом Монте-Карло
# -----------------------------------------------------

# Спосіб 1: метод середнього значення
N = 100000  # Кількість випадкових точок
# Генеруємо N випадкових точок на відрізку [a, b]
x_random = np.random.uniform(a, b, N)
# Обчислюємо середнє значення функції в цих точках
integral_MC = (b - a) * np.mean(f(x_random))
print("Результат інтегрування (метод Монте-Карло, середнє значення):", integral_MC)

# Спосіб 2: метод "hit or miss"
# Для цього методу нам необхідно визначити прямокутник, що містить область під графіком.
# Оскільки f(x)=x^2 на [0,2] є монотонно зростаючою, максимальне значення досягається при x = 2:
f_max = f(b)  # f(2) = 4
N_points = 10000  # Кількість випадкових точок для методу "hit or miss"

# Генеруємо випадкові точки (x, y) у прямокутнику [a, b] x [0, f_max]
x_rand = np.random.uniform(a, b, N_points)
y_rand = np.random.uniform(0, f_max, N_points)

# Визначаємо, які точки потрапляють під графік функції
points_under_curve = y_rand <= f(x_rand)
# Площа прямокутника: (b - a) * f_max
integral_MC_hitmiss = (b - a) * f_max * np.sum(points_under_curve) / N_points
print("Результат інтегрування (метод Монте-Карло, hit or miss):", integral_MC_hitmiss)

# -----------------------------------------------------
# 3. Перевірка розрахунків за допомогою quad та аналітичного підрахунку
# -----------------------------------------------------

# Обчислення інтегралу за допомогою SciPy (функція quad)
integral_quad, error_quad = quad(f, a, b)
print("Результат інтегрування (quad):", integral_quad)
print("Оціночна похибка quad:", error_quad)

# Аналітичний розрахунок інтегралу:
# ∫[0,2] x^2 dx = [x^3/3]_0^2 = 2^3/3 = 8/3 ≈ 2.66667
integral_exact = (b ** 3) / 3
print("Аналітичний результат інтегралу:", integral_exact)
