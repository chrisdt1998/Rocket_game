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

    def move(self, keys):
        # if keys[pygame.K_w] or keys[pygame.K_UP]:
        #     if self.speed + self.size <= self.position[1]:
        #         self.position[1] -= self.speed
        # elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        #     if self.position[1] <= 500 - self.speed - self.size:
        #         self.position[1] += self.speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            for block in self.blocks:
                if block[0] > 0:
                    block[0] -= 1
                else:
                    break
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            for block in reversed(self.blocks):
                if block[0] < 19:
                    block[0] += 1
                else:
                    break

    def update_board(self, board):
        for block in self.blocks:
            if board[block[0], block[1]] == 1:
                print('game over')
                game.running = False
            board[block[0], block[1]] = 2
        return board

    def bullet_movement(self):
        for bullet in self.bullets_shot:
            bullet.move()
            if bullet.position[1] < 0:
                self.bullets_shot.pop(self.bullets_shot.index(bullet))


class Bullets(object):
    def __init__(self, bullet_settings, position, window):
        self.bullet_settings = bullet_settings
        self.position = position
        self.window = window

    def move(self):
        self.position[1] -= self.bullet_settings['speed']

    def draw(self):
        pygame.draw.circle(self.window, self.bullet_settings['colour'], self.position, self.bullet_settings['size'])

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
    def __init__(self, grid_colour=(255, 255, 255), rock_colour=(255, 0, 0), player_colour=(0, 255, 0), bullet_colour=(0, 0, 255), dimensions=(20, 30), square_size=20):
        self.grid_colour = grid_colour
        self.rock_colour = rock_colour
        self.player_colour = player_colour
        self.bullet_colour = bullet_colour
        self.window = pygame.display.set_mode((dimensions[0] * square_size, dimensions[1] * square_size))
        self.dimensions = dimensions
        self.square_size = square_size
        self.running = True
        self.reset()

    def reset(self):
        self.player = Player(1)
        self.board = self.reset_board()

        self.rocks = []
        # speed is considered as time per block
        self.rock_base_speed = 1
        self.speed_bound = 0.1
        # self.speed_timer = 0
        self.rock_spawn_timer = 0
        self.frame_iteration = 0

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
        for rock in rocks_to_remove:
            self.rocks.pop(self.rocks.index(rock))
    
    def move_player(self, key):
        self.player.move(key)

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

    def run(self):
        while self.running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                self.reset()
            if keys[pygame.K_q]:
                self.running = False
            self.move_player(keys)
            self.move_rocks()
            self.rock_timer()
            self.board = self.update_board()
            # self.player.bullet_movement()

            self.update_screen()

        pygame.quit()


def get_square(value, square_size):
    if value % square_size > square_size / 2:
        return value + (square_size - (value % square_size))
    else:
        return value - (value % square_size)


clock = pygame.time.Clock()
speed = 10

game = Game()
game.run()

