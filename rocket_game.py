import pygame
import numpy as np
import math

pygame.init()

class Player(object):
    def __init__(self, size, colour, window, speed, bullet_speed, bullet_size, bullet_colour):
        self.size = size
        self.colour = colour
        self.position = [250, 450]
        self.window = window
        self.speed = speed
        self.bullets = []
        self.bullet_speed = bullet_speed
        self.bullet_size = bullet_size
        self.bullet_colour = bullet_colour

    def draw(self):
        pygame.draw.rect(self.window, self.colour, (self.position[0], self.position[1] - self.size, self.size, self.size))
        pygame.draw.rect(self.window, self.colour, (self.position[0] - self.size, self.position[1], self.size * 3, self.size))

    def move(self, keys):
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            if self.speed + self.size <= self.position[1]:
                self.position[1] -= self.speed
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            if self.position[1] <= 500 - self.speed - self.size:
                self.position[1] += self.speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            if self.speed + self.size <= self.position[0]:
                self.position[0] -= self.speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            if self.position[0] <= 500 - self.speed - (self.size * 2):
                self.position[0] += self.speed
        if keys[pygame.K_SPACE]:
            if len(self.bullets) < 10:
                self.bullets.append(Bullets(self.bullet_colour, self.bullet_size, self.bullet_speed, self.position.copy(), self.window))

    def bullet_movement(self):
        for bullet in self.bullets:
            bullet.move()
            if bullet.position[1] < 0:
                self.bullets.pop(self.bullets.index(bullet))


class Bullets(object):
    def __init__(self, colour, size, speed, position, window):
        self.colour = colour
        self.size = size
        self.speed = speed
        self.position = position
        self.window = window

    def move(self):
        self.position[1] -= self.speed

    def draw(self):
        pygame.draw.circle(self.window, self.colour, self.position, self.size)

class Rock(object):
    def __init__(self, size, colour, speed, position, window):
        self.size = size
        self.colour = colour
        self.position = position
        self.speed = speed
        self.window = window

    def move(self):
        self.position[1] += self.speed

    def draw(self):
        pygame.draw.circle(self.window, self.colour, self.position, self.size)


class Game(object):
    def __init__(self, background_colour, window, player, rock_colour, rock_speed):
        self.background_colour = background_colour
        self.window = window
        self.running = True
        self.player = player
        self.rocks = []
        self.rock_colour = rock_colour
        self.rock_speed = rock_speed
        self.rock_radius = 1

    def update_screen(self):
        self.window.fill(self.background_colour)
        self.player.draw()
        for rock in self.rocks:
            rock.draw()
        for bullet in self.player.bullets:
            bullet.draw()
        pygame.display.update()

    def rock_timer(self):
        if self.rock_radius > 0:
            self.rock_radius -= self.rock_speed
        else:
            self.rock_spawn()

    def rock_spawn(self):
        size = np.random.randint(5, 20)
        position = [np.random.randint(0 + size, 500 - size), size]
        self.rock_radius = size * 2
        rock = Rock(size, self.rock_colour, self.rock_speed, position, self.window)
        self.rocks.append(rock)

    def rock_movement(self):
        for rock in self.rocks:
            rock.move()
            if rock.position[1] - rock.size >= 500:
                self.rocks.pop(self.rocks.index(rock))

    def check_collision(self):
        for bullet in self.player.bullets:
            for rock in self.rocks:
                x = bullet.position[0] - rock.position[0]
                y = bullet.position[1] - rock.position[1]
                c = math.sqrt(x ** 2 + y ** 2)
                if c <= bullet.size + rock.size:
                    self.player.bullets.pop(self.player.bullets.index(bullet))
                    self.rocks.pop(self.rocks.index(rock))

        for rock in self.rocks:
            player_size = self.player.size
            corners_x = [player_size, 0, player_size, 2 * player_size]
            corners_y = [0, player_size, player_size, 0]
            corners_x = [x + self.player.position[0] for x in corners_x]
            corners_y = [y + self.player.position[1] for y in corners_y]
            for x, y in zip(corners_x, corners_y):
                x -= rock.position[0]
                y -= rock.position[1]
                c = math.sqrt(x ** 2 + y ** 2)
                if c <= rock.size:
                    print(f"Corner hit!")
                    self.running = False
                    return
            if self.player.position[1] - player_size >= rock.position[1] >= self.player.position[1] + player_size:
                if self.player.position[0] <= rock.position[0] <= self.player.position[0] + player_size:
                    print(f"Top hit!")
                    self.running = False
                    return

            if self.player.position[1] - player_size >= rock.position[1] >= self.player.position[1]:
                if self.player.position[0] - player_size <= rock.position[0] <= self.player.position[0] + (2 * player_size):
                    print(f"Side hit!")
                    self.running = False
                    return

    def run(self):
        while self.running:
            clock.tick(27)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            keys = pygame.key.get_pressed()
            player.move(keys)
            self.rock_movement()
            self.rock_timer()
            self.player.bullet_movement()
            self.check_collision()

            self.update_screen()

        pygame.quit()


clock = pygame.time.Clock()
window_size = (500, 500)
window = pygame.display.set_mode(window_size)
speed = 10
player = Player(10, (0, 255, 0), window, speed, speed * 1.5, 5, (125, 0, 125))
game = Game((0, 0, 0), window, player, (255, 0, 0), 5)
game.run()