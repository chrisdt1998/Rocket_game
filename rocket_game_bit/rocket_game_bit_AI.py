"""
This file represents the rocket game which can be played via human input.

This file was created and designed by Christopher du Toit.
"""

import pygame
import numpy as np

pygame.init()

class Player(object):
    """
    Class representing the player object.
    """
    def __init__(self, speed):
        self.speed = speed
        self.blocks = [[9, 28], [10, 28], [10, 27], [11, 28]]
        self.is_alive = True

    def move(self, action):
        if action[0] == 1:
            for block in self.blocks:
                if block[0] > 0:
                    block[0] -= 1
                else:
                    break
        if action[1] == 1:
            for block in reversed(self.blocks):
                if block[0] < 19:
                    block[0] += 1
                else:
                    break

    def update_board(self, board):
        for block in self.blocks:
            if board[block[0], block[1]] == 1:
                self.is_alive = False
                return board
            board[block[0], block[1]] = 2
        return board

class Rock(object):
    def __init__(self, colour, speed, blocks):
        self.colour = colour
        self.blocks = blocks
        self.speed = speed
        self.time_in_block = 0

    def move(self):
        self.time_in_block += 1
        if self.time_in_block >= 1 / self.speed:
            self.time_in_block = 0
            for block in self.blocks:
                block[1] += 1
                if block[1] >= 30:
                    return False
        return True

    def update_board(self, board):
        for block in self.blocks:
            board[block[0], block[1]] = 1
        return board


class Game(object):
    def __init__(self, grid_colour=(255, 255, 255), rock_colour=(255, 0, 0), player_colour=(0, 255, 0), bullet_colour=(0, 0, 255), dimensions=(20, 30), square_size=20, show_visuals=False):
        self.grid_colour = grid_colour
        self.rock_colour = rock_colour
        self.player_colour = player_colour
        self.bullet_colour = bullet_colour
        self.dimensions = dimensions
        self.square_size = square_size
        self.show_visuals = show_visuals
        self.repeated_moves = []
        self.reset()

    def reset(self):
        if self.show_visuals:
            self.clock = pygame.time.Clock()
            self.window = pygame.display.set_mode((self.dimensions[0] * self.square_size, self.dimensions[1] * self.square_size))
        else:
            self.window = None
        self.running = True
        self.player = Player(1)
        self.board = self.reset_board()
        self.score = 0
        self.rocks_survived = 0

        self.rocks = []
        # speed is considered as time per block
        self.rock_base_speed = 1
        self.speed_bound = 0.1
        # self.speed_timer = 0
        self.rock_spawn_timer = 0
        self.kickstart()
    
    def kickstart(self):
        for _ in range(15):
            self.move_player([0, 0])
            self.move_rocks()
            self.rock_timer()
            self.board = self.update_board()

    def draw_board(self):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                # Empty square
                if self.board[i, j] == 0:
                    pygame.draw.rect(self.window, self.grid_colour, (i * self.square_size, j * self.square_size, self.square_size, self.square_size), 1)
                # Rock
                if self.board[i, j] == 1:
                    pygame.draw.rect(self.window, self.rock_colour, (i * self.square_size, j * self.square_size, self.square_size, self.square_size))
                # Player
                if self.board[i, j] == 2:
                    pygame.draw.rect(self.window, self.player_colour, (i * self.square_size, j * self.square_size, self.square_size, self.square_size))
                # if self.board[i, j] == 3:
                #     pygame.draw.rect(self.window, self.bullet_colour, (i * self.square_size, j * self.square_size, self.square_size, self.square_size))

    def reset_board(self):
        return np.zeros(self.dimensions)

    def move_rocks(self):
        rocks_to_remove = []
        for rock in self.rocks:
            rock_moved = rock.move()
            if not rock_moved:
                rocks_to_remove.append(rock)
        self.rocks_survived += len(rocks_to_remove)
        for rock in rocks_to_remove:
            self.rocks.pop(self.rocks.index(rock))

    def move_player(self, action):
        if len(self.repeated_moves) == 0:
            self.repeated_moves.append(action)
        elif len(self.repeated_moves) == 1 and action != self.repeated_moves[0]:
            self.repeated_moves.append(action)
        elif action in self.repeated_moves and action != self.repeated_moves[-1]:
            self.repeated_moves.append(action)
        else:
            self.repeated_moves = [action]
        self.player.move(action)

    def update_board(self):
        board = self.reset_board()
        for rock in self.rocks:
            board = rock.update_board(board)
        board = self.player.update_board(board)
        return board

    def update_screen(self):
        self.window.fill((0, 0, 0))
        self.draw_board()
        pygame.display.update()
        # breakpoint()

    def rock_timer(self):
        if self.rock_spawn_timer > 0:
            self.rock_spawn_timer -= 1
        else:
            self.rock_spawn()

    def rock_spawn(self):
        position = [np.random.randint(0, self.dimensions[0]), 0]
        speed_bound = min(self.speed_bound, 0.5)
        rock_speed = max(1, np.random.randint(self.rock_base_speed * (1 - speed_bound), self.rock_base_speed * (1 + speed_bound)))
        self.speed_bound += 0.01
        self.rock_base_speed += 0.01
        self.rock_spawn_timer = 1 / rock_speed

        rock = Rock(self.rock_colour, rock_speed, [position])
        self.rocks.append(rock)

    def check_danger_zone(self):
        score = 0
        checked_locations = []
        for block in self.player.blocks:
            for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                x_coord = block[0] + x
                y_coord = block[1] + y
                if x_coord < 0 or x_coord >= self.dimensions[0]:
                    continue
                if y_coord < 0 or y_coord >= self.dimensions[1]:
                    continue

                if self.board[x_coord][y_coord] == 1 and (x_coord, y_coord) not in checked_locations and [x_coord, y_coord] not in self.player.blocks:
                    checked_locations.append((x_coord, y_coord))
                    score -= 1
        return score

    def play_step(self, action):
        if self.show_visuals:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self.move_player(action)
        self.move_rocks()
        self.rock_timer()
        self.board = self.update_board()

        reward = 0
        reward = self.check_danger_zone()
        done = False
        if not self.player.is_alive:
            reward -= -10
            done = True
            return reward, done, self.score

        reward += self.rocks_survived
        if len(self.repeated_moves) > 2:
            reward -= ((len(self.repeated_moves) - 2) / 10) * 2
        self.score += reward
        self.rocks_survived = 0

        if self.show_visuals:
            self.update_screen()
            self.clock.tick(30)
        return reward, done, self.score
