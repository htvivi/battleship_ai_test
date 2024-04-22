# 0 : ismeretlen cellák
# 1 : hajót tartalmazó cella
# 2 : találat
# 3 : mellé
# 4 : elsüllyedt hajó
# 5 : ismert hajóhoz közeli találat

import pygame
import random
import torch
import numpy as np
from training import DQN

# Pygame inicializálása
pygame.init()
pygame.display.set_caption("BattleShip with AI")
pygame.font.init()

# Ablak előkészítése, színek
WIDTH = 900
HEIGHT = 800
BLUE = (31,47,69) # A háttér színe
BACKGROUND = (56,93,141) # Tábla színe
SHIPS = (207, 193, 189) # Hajók színe
WHITE = (255, 255, 255) # Tábla körvonala
BLACK = (0, 0, 0)
GREEN = (134,239,172)
RED =  (239, 68, 68)
ORANGE = (253,186,116)
COLORS = {'0': BLUE, '2': ORANGE, '3': RED, '4': GREEN}
display = pygame.display.set_mode((WIDTH, HEIGHT))

# Táblához kapcsolódó értékek
CELLSIZE = 30
ROWS = 10
COLS = 10

class Ship:
    def __init__(self, size):
        self.x = random.randrange(0, 9) # Sor koordináta
        self.y = random.randrange(0, 9) # Oszlop koordináta
        self.size = size
        self.position = random.choice(["horizontal", "vertical"]) # Vízszintes vagy függőleges pozíciót vesz fel
        self.coordinates = self.shipCoordinates()

    def shipCoordinates(self):
        # Hajó koordináták kiszámolása + pozíció
        firstCoord = self.y * 10 + self.x
        if self.position == 'horizontal':
            return [firstCoord + i for i in range(self.size)]
        elif self.position == 'vertical':
            return [firstCoord + i * 10 for i in range(self.size)]
        
class Player:
    def __init__(self):
        self.ships = [] # Játékos hajói
        self.ocean = ["0" for i in range(100)] # Alapértelmezetten feltöltjük 100 db 0-val (ocean) a listát
        shipSize = [5, 4, 3, 2, 2] # Játékos hajóinak mérete
        self.shipsOnBoard(shipSize)
        coordInList = [ship.coordinates for ship in self.ships]
        self.shipsList = [i for l in coordInList for i in l]
        self.sunkShips = [] # Játékos elsüllyesztett hajói

    def shipsOnBoard(self, shipSize):
        # Hajók felhelyezése a táblára random módon
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
        # CSAK ELLENŐRZÉSHEZ !!! - hajók kirajzolása a táblára a terminalon
        coordinates = ["0" if i not in self.shipsList else "1" for i in range(100)]
        for x in range(10):
            print(" ".join(coordinates[(x-1)*10:x*10]))

class BattleShip:
    def __init__(self, model_path):
        self.player1 = Player()
        self.player2 = BattleShipAI(model_path)
        self.player1_turn = True
        self.gameOver = False
        # Játékos és AI lépéseinek mentése halmazba
        self.taken_actions_player = set()
        self.taken_actions_opponent = set()

    def playersTurn(self, i):
        player = self.player1 if self.player1_turn else self.player2
        opponent = self.player2 if self.player1_turn else self.player1
        hit = False

        # Ha korábban választott lépés kerül kiválasztásra, terminalon jelez és újra próbálkozhat
        if i in (self.taken_actions_player if self.player1_turn else self.taken_actions_opponent):
            print(f"Cell {i} has been already chosen. Try again!")
            return

        if self.player1_turn:
            self.taken_actions_player.add(i)
        else:
            self.taken_actions_opponent.add(i)

        if i in opponent.shipsList:
            player.ocean[i] = "2"
            hit = True
            sunk_ship = self.checkSunkShips(player, opponent)
            if sunk_ship:
                self.updateSunk(player.ocean, sunk_ship)
        else:
            player.ocean[i] = "3"

        if self.checkAllShipsSunk(opponent):
            self.gameOver = True
            self.result = 'Player' if self.player1_turn else 'AI'

        # Ha talált az utolsó lövés, akkor újra a játékos vagy AI következik
        if not hit:
            self.player1_turn = not self.player1_turn

    def updateSunk(self, ocean, ship):
        # Frissíti az elsüllyesztett hajót
        for coord in ship.coordinates:
            ocean[coord] = "4"

    def checkSunkShips(self, player, opponent):
        # Elsüllyedt a hajó?
        for ship in opponent.ships:
            if ship not in opponent.sunkShips and all(player.ocean[coord] == "2" for coord in ship.coordinates):
                opponent.sunkShips.append(ship)
                return ship
        return None

    def checkAllShipsSunk(self, player):
        # Elsüllyedt az összes hajó? -> Játék vége
        return len(player.sunkShips) == len(player.ships)

    def printFullGrid(self, ocean):
        # CSAK ELLENŐRZÉSHEZ !!! - Játéktábla kirajzolása terminalra
        print("Board:")
        for row in range(10):
            for col in range(10):
                cell = ocean[row * 10 + col]
                print(cell, end=" ")
            print()

    def getState(self):
        # A játéktábla átalakítása, hogy feldolgozható legyen az AI számára
        state_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        state = [state_mapping[cell] for cell in self.player2.ocean]

        state = np.array(state, dtype=np.float32)

        state = state.flatten()

        return state

class Button:
	def __init__(self, img, pos, input, font, color, hover_color):
		self.img = img
		self.x = pos[0]
		self.y = pos[1]
		self.font = font
		self.color, self.hover_color = color, hover_color
		self.input = input
		self.text = self.font.render(self.input, True, self.color)
		self.rect = self.img.get_rect(center=(self.x, self.y))
		self.text_rect = self.text.get_rect(center=(self.x, self.y))

	def updateButton(self, display):
        # Gomb és szövegének blit-elése a képernyőre
		display.blit(self.img, self.rect)
		display.blit(self.text, self.text_rect)

	def buttonClick(self, pos):
        # A gombra kattintáskor True értéket ad vissza
		if pos[0] in range(self.rect.left, self.rect.right) and pos[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def hoverColor(self, pos):
        # Megváltoztatja a szöveg színét, ha ráhúzzuk az egeret
		if pos[0] in range(self.rect.left, self.rect.right) and pos[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.input, True, self.hover_color)
		else:
			self.text = self.font.render(self.input, True, self.color)

class BattleShipAI(Player):
    def __init__(self, model_path):
        super(BattleShipAI, self).__init__()
        # Betanított modell betöltése
        self.model = DQN(100, 128, 100)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def selectAction(self, state, valid_actions):
        # Cselekvés kiválasztása a betöltött modell alapján
        state = [int(cell) for row in self.ocean for cell in row]
        state_tensor = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float32).flatten().unsqueeze(0)

        with torch.no_grad():
            predicted_q_values = self.model(state_tensor)

        masked_q_values = predicted_q_values.clone().squeeze()

        for i in range(len(masked_q_values)):
            if i not in valid_actions:
                masked_q_values[i] = float('-inf')

        action = torch.argmax(masked_q_values).item()

        return action

def drawBoard(player, marginLeft = 0, marginTop = 0, search = False):
    # Kirajzolja a játéktáblát a képernyőre
    for i in range(100):
        x = marginLeft + i % 10 * CELLSIZE
        y = marginTop + i // 10 * CELLSIZE
        cell = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(display, BLUE, cell)
        pygame.draw.rect(display, WHITE, cell, 1)
        if search:
            # Kattintáskor, ha "keresőtáblán" történik a kattintás, akkor színes négyzetek jelzik a lövés kimenetelét
            x += CELLSIZE // 2
            y += CELLSIZE // 2
            rect_width = CELLSIZE // 2
            rect_height = CELLSIZE // 2
            pygame.draw.rect(display, COLORS[player.ocean[i]], (x - rect_width // 2, y - rect_height // 2, rect_width, rect_height))

def drawShips(player, marginLeft = 0, marginTop = 0):
    # Hajók kirajzolása a képernyőre
    for ship in player.ships:
        x = marginLeft + ship.y * CELLSIZE + 7
        y = marginTop + ship.x * CELLSIZE + 7
        if ship.position == "horizontal":
            width = ship.size * CELLSIZE - 14
            height = CELLSIZE - 14
        elif ship.position == "vertical":
            width = CELLSIZE - 14
            height = ship.size * CELLSIZE - 14
        cell = pygame.Rect(x, y, width, height)
        pygame.draw.rect(display, SHIPS, cell)

def loadFont(fontsize):
    # Betűtípus a szöveg betöltéséhez
    return pygame.font.Font("assets/PixelifySans-VariableFont_wght.ttf", fontsize)

def game(model_path):
    # Játékablak
    # Játék példány létrehozása
    battleship = BattleShip(model_path)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Játék bezárása user inputra
                pygame.quit()

            if battleship.player1_turn:
                # Felhasználó következik
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if event.button == 1:
                        x1 = x - 100
                        y1 = y - 50

                        if 0 <= x1 < 10 * CELLSIZE and 0 <= y1 < 10 * CELLSIZE:
                            row = x1 // CELLSIZE
                            col = y1 // CELLSIZE
                            coord = col * 10 + row
                            battleship.playersTurn(coord)
                            print(coord)

            else:
                # AI következik
                state = battleship.getState()
                valid_actions = [i for i in range(100) if i not in battleship.taken_actions_opponent]
                action = battleship.player2.selectAction(state, valid_actions)
                battleship.playersTurn(action)
                print(action)

        # Kirajzolás képernyőre
        display.fill(BACKGROUND)
        drawBoard(battleship.player1, 100, 50, search = True)
        drawBoard(battleship.player2, 100, 450)
        drawBoard(battleship.player1, 500, 50)
        drawBoard(battleship.player2, 500, 450, search = True)

        drawShips(battleship.player1, 100, 450)
        # drawShips(battleship.player2, 500, 50)  # AI hajói

        # Játék vége
        if battleship.gameOver:
            text = battleship.result + ' wins! Press right mouse button to play again.'
            RESULT = loadFont(30).render(text, False, WHITE)
            RESULT_RECT = RESULT.get_rect(center=(450, 400))
            display.blit(RESULT, RESULT_RECT)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    battleship = BattleShip(model_path)

        pygame.display.flip()

def main_menu():
    # Főmenü
    while True:
        # Grafikus biszbasz
        display.fill(BACKGROUND)
        
        MENU_TEXT = loadFont(75).render("WELCOME TO BATTLESHIP", True, WHITE)

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_RECT = MENU_TEXT.get_rect(center=(450, 100))

        SHIP_IMG = pygame.image.load("assets/ship.png")

        SHIP_RECT = SHIP_IMG.get_rect(center=(450, 300))

        PLAY_BUTTON = Button(pygame.image.load("assets/button1.png"), (450, 500), "PLAY", loadFont(50), BLACK, RED)
        
        QUIT_BUTTON = Button(pygame.image.load("assets/button1.png"), (450, 650), "QUIT", loadFont(50), BLACK, RED)
        
        display.blit(MENU_TEXT, MENU_RECT)
        display.blit(SHIP_IMG, SHIP_RECT)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.hoverColor(MENU_MOUSE_POS)
            button.updateButton(display)

        pygame.display.update()
        
        for event in pygame.event.get():
            # Kilépés a játékből user inputra
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.buttonClick(MENU_MOUSE_POS):
                    # PLAY gombra kattintva játék indítása
                    game(model_path)
                if QUIT_BUTTON.buttonClick(MENU_MOUSE_POS):
                    # QUIT gombra kattintva kilépés
                    pygame.quit()

if __name__ == "__main__":
    model_path = 'models/butuska.pth' # Modell elérési útvonala
    main_menu() # Főmenü betöltése a játék indításakor