import pygame
import numpy as np
import math

pygame.init()

class Player(object):
    """
    Class representing the player object.
    """
    def __init__(self, size, colour, window, speed, bullet_speed, bullet_size, bullet_colour):
        self.player_settings = {'size': size, 'colour': colour, 'speed': speed}
        self.bullet_settings = {'speed': bullet_speed, 'size': bullet_size, 'colour': bullet_colour}
        self.bullets_shot = []
        self.position = [250, 450]
        self.window = window

    def draw(self):
        pygame.draw.rect(
            self.window,
            self.player_settings['colour'],
            (self.position[0] - self.player_settings['size'],
             self.position[1], self.player_settings['size'] * 3,
             self.player_settings['size'] * 3)
        )

    def move(self, keys):
        # if keys[pygame.K_w] or keys[pygame.K_UP]:
        #     if self.speed + self.size <= self.position[1]:
        #         self.position[1] -= self.speed
        # elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        #     if self.position[1] <= 500 - self.speed - self.size:
        #         self.position[1] += self.speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            if self.player_settings['speed'] + self.player_settings['size'] <= self.position[0]:
                self.position[0] -= self.player_settings['speed']
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            if self.position[0] <= 500 - self.player_settings['speed'] - (self.player_settings['size'] * 2):
                self.position[0] += self.player_settings['speed']
        if keys[pygame.K_SPACE]:
            if len(self.bullets_shot) < 10:
                self.bullets_shot.append(Bullets(self.bullet_settings, self.position.copy(), self.window))


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
    def __init__(self, background_colour, window, rock_colour):
        self.rock_colour = rock_colour
        self.background_colour = background_colour
        self.window = window
        self.running = True

        self.reset()

    def reset(self):
        self.player = Player(10, (0, 255, 0), window, speed, speed * 1.5, 5, (125, 0, 125))

        self.rocks = []
        self.rock_speed = 1
        self.speed_lwr_bnd = 1
        self.speed_upr_bnd = self.speed_lwr_bnd + 4
        self.rock_radius = 1
        self.speed_timer = 0
        self.frame_iteration = 0

    def update_screen(self):
        self.window.fill(self.background_colour)
        self.player.draw()
        for rock in self.rocks:
            rock.draw()
        for bullet in self.player.bullets_shot:
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
        self.rock_speed = np.random.randint(self.speed_lwr_bnd, self.speed_upr_bnd)
        self.rock_radius = size * 2
        rock = Rock(size, self.rock_colour, self.rock_speed, position, self.window)
        self.rocks.append(rock)

    def rock_movement(self):
        for rock in self.rocks:
            rock.move()
            if rock.position[1] - rock.size >= 500:
                self.rocks.pop(self.rocks.index(rock))

    def check_collision(self):
        for bullet in self.player.bullets_shot:
            for rock in self.rocks:
                x = bullet.position[0] - rock.position[0]
                y = bullet.position[1] - rock.position[1]
                c = math.sqrt(x ** 2 + y ** 2)
                if c <= bullet.bullet_settings['size'] + rock.size:
                    self.player.bullets_shot.pop(self.player.bullets_shot.index(bullet))
                    self.rocks.pop(self.rocks.index(rock))
                    return

        for rock in self.rocks:
            player_size = self.player.player_settings['size']
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

    def reward(self):
        pass



    def run(self):
        while self.running:
            clock.tick(27)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                self.reset()
            self.player.move(keys)
            self.rock_movement()
            self.rock_timer()
            self.player.bullet_movement()
            self.check_collision()

            self.update_screen()
            if pygame.time.get_ticks()/1000 > self.speed_timer:
                print(f"Speed increase!")
                self.speed_timer += 1
                self.speed_lwr_bnd += 0.1
                self.speed_upr_bnd += 0.1

        pygame.quit()


clock = pygame.time.Clock()
window_size = (500, 500)
window = pygame.display.set_mode(window_size)
speed = 10

game = Game((0, 0, 0), window, (255, 0, 0))
game.run()