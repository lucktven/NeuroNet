# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Перевірка, чи існує файл з результатами
if not os.path.exists("results/data.csv"):
    raise FileNotFoundError("Файл results/data.csv не знайдено. Запустіть main.py для генерації результатів.")

# Завантаження результатів
df = pd.read_csv("results/data.csv")

# Графік: час виконання vs кількість агентів
os.makedirs("results/graphs", exist_ok=True)  # Створення папки для графіків
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x="num_agents", y="duration", marker="o")
plt.title("Час виконання залежно від кількості агентів")
plt.xlabel("Кількість агентів")
plt.ylabel("Час виконання (секунди)")
plt.grid()
plt.savefig("results/graphs/time_vs_agents.png")
plt.show()

# Графік: знайдені ресурси vs кількість агентів
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x="num_agents", y="total_resources_found")
plt.title("Знайдені ресурси залежно від кількості агентів")
plt.xlabel("Кількість агентів")
plt.ylabel("Кількість знайдених ресурсів")
plt.grid()
plt.savefig("results/graphs/resources_vs_agents.png")
plt.show()
