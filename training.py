import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
# from google.colab import drive

# drive.mount('/content/drive') # ---> csak Google Colabhoz !!!

# Környezet a modell tanításához
class Ship:
    def __init__(self, size):
        # Random hajó kezdőkoordináta + pozíció
        self.x = random.randrange(0, 9)
        self.y = random.randrange(0, 9)
        self.size = size
        self.position = random.choice(["horizontal", "vertical"])
        self.coordinates = self.shipCoordinates()

    def shipCoordinates(self):
        # Hajókoordináták megállapítása
        firstCoord = self.y * 10 + self.x
        if self.position == 'horizontal':
            return np.arange(firstCoord, firstCoord + self.size)
        elif self.position == 'vertical':
            return np.arange(firstCoord, firstCoord + self.size * 10, 10)

class Player:
    def __init__(self):
        self.ships = []
        self.ocean = np.full(100, "0")
        shipSize = [5, 4, 3, 2, 2]
        self.shipsOnBoard(shipSize)
        coordInList = [ship.coordinates for ship in self.ships]
        self.shipsList = np.concatenate(coordInList)
        self.sunkShips = []

    def shipsOnBoard(self, shipSize):
        # Hajók felhelyezése a táblára
        for s in shipSize:
            placed = False
            while not placed:
                ship = Ship(s)

                canBePlaced = True
                for i in ship.coordinates:
                    if i >= 100:
                        canBePlaced = False
                        break

                    for placedShip in self.ships:
                        if i in placedShip.coordinates:
                            canBePlaced = False
                            break

                    new_x = i // 10
                    new_y = i % 10
                    if new_x != ship.x and new_y != ship.y:
                        canBePlaced = False
                        break

                if canBePlaced:
                    self.ships.append(ship)
                    placed = True

    def printShips(self):
        coordinates = np.where(np.isin(np.arange(100), self.shipsList), "1", "0")
        for x in range(10):
            print(" ".join(coordinates[(x-1)*10:x*10]))

class BattleShipAI(Player):
    def __init__(self):
        super(BattleShipAI, self).__init__()

    def getState(self, state):
        # Tábla állapota agentnek
        self.state = state

class BattleShip:
    def __init__(self) :
        self.player1 = BattleShipAI()
        self.player2 = Player()
        self.player_turn = True
        self.gameOver = False
        self.state = np.full(100, "0")
        self.sunkShips = []
        self.total_rewards = []
        self.chosen_cells = set()
        self.last_hit_step = 0
        self.step_count = 0
        self.max_steps = 100

    def playersTurn(self, i):
        # Ki következik? + reward
        player = self.player1 if self.player_turn else self.player2
        opponent = self.player2 if self.player_turn else self.player1
        reward = 0

        if self.player_turn:
            self.chosen_cells.add(i)
            self.step_count += 1

        if i in opponent.shipsList:
            player.ocean[i] = "2"
            self.last_hit_step = 0

            sunk_ship = self.checkSunkShips(player, opponent)
            if sunk_ship:
                self.updateSunk(player.ocean, sunk_ship)
                if self.player_turn:
                    reward += 12

            if self.player_turn:
                reward += 6
        else:
            player.ocean[i] = "3"
            if self.player_turn:
                reward -= 0.2
                self.last_hit_step += 1

        if self.player_turn:
            reward -= 0.2 * self.last_hit_step

        if self.checkAllShipsSunk(opponent):
            if self.player_turn:
                reward += 20
            else:
                reward -= 20
            self.gameOver = True

        if self.step_count >= self.max_steps:
            self.gameOver = True
            reward -= 10
            return self.state, reward, self.gameOver

        if self.player_turn:
            self.total_rewards.append(reward)

        self.player_turn = not self.player_turn

        return self.state, reward, self.gameOver

    def updateSunk(self, ocean, ship):
        # Hajó elsüllyedésekor tábla frissítése
        for coord in ship.coordinates:
            ocean[coord] = "4"

    def printBoard(self, ocean):
        # ELLENŐRZÉSHEZ !!!
        print("Board:")
        for row in range(10):
            for col in range(10):
                cell = ocean[row * 10 + col]
                print(cell, end=" ")
            print()

    def randomOpponent(self):
        # Random lövő
        search = self.player1.ocean if self.player_turn else self.player2.ocean
        unknown = np.where(search == '0')[0]

        if unknown.size > 0:
            random_coord = random.choice(unknown)
        else:
            random_coord = None

    def checkSunkShips(self, player, opponent):
        # Elsüllyedt az adott hajó?
        for ship in opponent.ships:
            if ship not in opponent.sunkShips and all(player.ocean[coord] == "2" for coord in ship.coordinates):
                opponent.sunkShips.append(ship)
                return ship
        return None

    def checkAllShipsSunk(self, player):
        # Összes hajó elsüllyedt?
        return len(player.sunkShips) == len(player.ships)

    def resetGame(self):
        # Epizód befejezésével a környezet alaphelyzetre hozása
        self.player1 = BattleShipAI()
        self.player2 = Player()
        self.player_turn = True
        self.gameOver = False
        self.state = np.full(100, "0")
        self.total_rewards = []
        self.chosen_cells = set()
        self.last_hit_step = 0
        self.step_count = 0

        return self.state.flatten()

    def maxSteps(self, episode):
        # Maximális lépés limit
        if episode < 20000:
            self.max_steps = 100
        elif episode < 40000:
            self.max_steps = 75
        elif episode < 100000:
            self.max_steps = 60
        else:
            self.max_steps = 50

class DQN(nn.Module):
    # Deep Q neurális hálók
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Automatikusan meghívódik és számol
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_path='/content/drive/My Drive/virus.pth'): # Fájl helye Google Driveon
        # Modell mentéséhez szükséges
        torch.save(self.state_dict(), file_path)

class Trainer:
    def __init__(self, model, lr, gamma):
        # A tanításhoz szükséges paraméterek
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.replay_memory = []
        self.MAX_MEMORY_SIZE = 10000
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.losses = []
        self.episodes = []

    def trainStep(self):
        # A háló tanításához
        self.optimizer.zero_grad()
        batch = random.sample(self.replay_memory, k=min(32, len(self.replay_memory)))
        states, rewards, next_states, _ = zip(*batch)
        states = [np.array(state, dtype=np.float32) for state in states]
        states = [torch.tensor(state) for state in states]
        states = torch.stack(states)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        rewards = torch.tensor(rewards)
        next_states = np.array(next_states, dtype=np.float32)
        next_states = torch.tensor(next_states)

        q_values = self.model(states)
        q_values_next = self.model(next_states)

        # Q értékek újraszámolása a Bellman egyenlettel
        expected_q_values = rewards + self.gamma * torch.max(q_values_next, dim=1, keepdim=True)[0]

        # Loss újraszámolása
        loss = self.loss_fn(q_values, expected_q_values)
        self.losses.append(loss.item())

        loss.backward()
        self.optimizer.step()

        return q_values, q_values_next

    def remember(self, state, reward, next_state, done):
        # Memóriából kiszórni a régi tapasztalatokat
        if len(self.replay_memory) >= self.MAX_MEMORY_SIZE:
            self.replay_memory.pop(0)
        self.replay_memory.append((state, reward, next_state, done))

BOARDSIZE = 10
LR = 0.000001 # Learning rate
BATCH_SIZE = 10000

class BattleShipAgent:
    def __init__(self):
        # Agent tanításához szükséges paraméterek
        self.board_size = BOARDSIZE
        self.model = DQN(self.board_size * self.board_size, 128, self.board_size * self.board_size)
        self.gamma = 0.99
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)
        self.epsilon = 1
        self.epsilon_decay = 0.99998
        self.min_epsilon = 0.1

    def selectAction(self, state, chosen_cells):
        # Cselekvés kiválasztása a legnagyobb Q érték alapján
        state_tensor = torch.tensor(np.array(state).astype(np.float32), dtype=torch.float32).flatten()

        with torch.no_grad():
            action_probs = self.model(state_tensor)

        # A választott cselekvések Q értékének lecsökkentése, hogy biztosan ne kerüljenek kiválasztásra
        mask = np.ones(action_probs.shape, dtype=bool)
        mask[list(chosen_cells)] = False

        action_probs_np = action_probs.detach().numpy()
        action_probs_np[~mask] = -1e8

        available_actions = np.where(mask)[0]
        if available_actions.size == 0:
            return None

        # Véletlenszerű cselekvés vagy modell által választott cselekvés
        if random.random() < self.epsilon:
            available_actions = np.where(mask)[0]
            action = np.random.choice(available_actions)
        else:
            action = np.argmax(action_probs_np)

        return action

    def decayEpsilon(self):
        # Epsilon érték csökkentése
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def training():
    # Tanító loop
    env = BattleShip()
    agent = BattleShipAgent()

    mean_losses = []
    epsilon_history = []

    # Jutalmak az összes epizód során
    all_episodes_rewards = []

    rewards_history = []

    num_episodes = 200000
    for episode in range(num_episodes):
        state = env.resetGame()
        gameOver = False
        total_rewards = []
        env.maxSteps(episode)

        # Jutalmak összegyűjtéséhez, egy adott epizódban !!!
        episode_rewards = []

        epsilon_history.append(agent.epsilon)

        print(f"Episode {episode + 1}:")

        while not gameOver:
            action = agent.selectAction(state, env.chosen_cells)
            if action is not None:
                next_state, reward, gameOver = env.playersTurn(action)
                if next_state is not None:
                    reward = float(reward)
                    agent.trainer.remember(state, reward, next_state, gameOver)
                    state = next_state
                    total_rewards.append(reward)
                    episode_rewards.append(reward)
                else:
                    print("Next state is None.")
                    break
            else:
                print("No valid action chosen.")
                break

            env.randomOpponent()

        rewards_history.append(sum(episode_rewards))
        total_reward = sum(total_rewards)
        all_episodes_rewards.append(total_reward)

        episode_losses = agent.trainer.losses[-len(total_rewards):] if total_rewards else [0]
        mean_loss = np.mean(episode_losses)
        mean_losses.append(mean_loss)

        if env.player_turn:
            print("AI wins!")
        else:
            print("Random wins!")

        agent.decayEpsilon()

        if len(agent.trainer.replay_memory) >= BATCH_SIZE:
            q_values, q_values_next = agent.trainer.trainStep()
            # Q értékek kiíratása a terminalra ellenőrzéshez
            print("Q-values:", q_values)
            print("Next Q-values:", q_values_next)

        total_rewards_per_episode = [sum(episode) if isinstance(episode, list) else episode for episode in rewards_history]

    # agent.model.save() # Mentéshez !!!

    # Minden 100. meccs kiíratása ...
    indices = list(range(0, len(total_rewards_per_episode), 100))
    selected_rewards = [total_rewards_per_episode[i] for i in indices]

    # Szemléltetés matplotlib-bel
    plt.figure(figsize=(15, 5))
    plt.scatter(indices, selected_rewards, label='Összes jutalom', alpha=0.6, edgecolors='w', s=100)
    plt.title('Összes jutalom 100 epizódonként')
    plt.xlabel('Epizód')
    plt.ylabel('Jutalom')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(mean_losses)
    plt.title('Átlagos hiba nagysága epizódonként')
    plt.xlabel('Epizód')
    plt.ylabel('Átlagos hiba')

    plt.figure(figsize=(10, 4))
    plt.plot(epsilon_history)
    plt.title('Epszilon csökkenése epizódonként')
    plt.xlabel('Epizód')
    plt.ylabel('Epszilon')
    plt.show()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    training()