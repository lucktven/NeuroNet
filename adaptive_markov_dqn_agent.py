!pip install numpy tensorflow matplotlib
import numpy as np
import tensorflow as tf
from collections import deque
import random
import csv
import matplotlib.pyplot as plt
class MarkovModel:
    def __init__(self, states):
        self.transition_matrix = np.full((states, states), 0.1)  # Початкова ймовірність переходів
        self.drift_threshold = 0.1  # Поріг для виявлення дрейфу
        self.adapt_rate = 0.5

    def update_transition_matrix(self, new_matrix):
        """Оновлення матриці переходів з новими ймовірностями."""
        self.transition_matrix = (self.adapt_rate * self.transition_matrix +
                                  (1 - self.adapt_rate) * new_matrix)

    def detect_drift(self, new_matrix):
        """Перевірка наявності дрейфу між старою та новою моделлю."""
        drift_score = np.sum(np.abs(self.transition_matrix - new_matrix))
        return drift_score > self.drift_threshold
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Знижуючий коефіцієнт
        self.epsilon = 1.0  # Дослідницька ймовірність
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        """Створення нейронної мережі для DQN."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Запам'ятовування досвіду."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Вибір дії на основі поточного стану."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Випадковий вибір
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Дія з найбільшою ймовірністю

    def replay(self, batch_size):
        """Навчання моделі на основі запам'ятованих досвідів."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
# Основна логіка програми
if __name__ == "__main__":
    states = 3  # Кількість станів
    model = MarkovModel(states)
    dqn_agent = DQNAgent(state_size=states, action_size=states)

    new_transitions = np.array([[0.2, 0.3, 0.5],
                                 [0.1, 0.7, 0.2],
                                 [0.3, 0.3, 0.4]])
# Перевірка на дрейф
if model.detect_drift(new_transitions):
        print("Concept drift detected. Adapting model...")
        model.update_transition_matrix(new_transitions)
else:
        print("No drift detected. Model remains unchanged.")
# Виконати навчання DQN
episodes = 1000
rewards_per_episode = []
transition_matrices = []  # Для збереження зміни матриці переходів

for e in range(episodes):  # Кількість епох
        state = np.random.rand(1, states)  # Випадковий початковий стан
        total_reward = 0

for time in range(500):  # Кількість кроків у кожній епосі
            action = dqn_agent.act(state)
            next_state = state  # Це слід змінити на основі дій
            reward = 1  # Нагорода слід визначити
            done = False  # Сигнал про завершення епохи
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

dqn_agent.replay(32)  # Навчання на основі пам'яті
rewards_per_episode.append(total_reward)
transition_matrices.append(model.transition_matrix.copy())  # Зберегти поточну матрицю
print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {dqn_agent.epsilon}")
# Збереження результатів у файл
with open('training_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward"])
        for i, reward in enumerate(rewards_per_episode):
            writer.writerow([i + 1, reward])
# Вивід остаточної матриці переходів
print("Final Transition Matrix:")
print(model.transition_matrix)
# Побудова графіків
    # Графік прогресу навчання
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
# Графік зміни матриці переходів
plt.subplot(1, 2, 2)
for i in range(states):
    plt.plot([mat[i].tolist() for mat in transition_matrices], label=f'State {i}')
plt.title('Transition Matrix Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Probability')
plt.legend()

plt.tight_layout()
plt.show()
 # Завантаження файлу з результатами
from google.colab import files
files.download('training_results.csv')
