import pygame
import numpy as np
import math

pygame.init()

class Player(object):
    def __init__(self, size, colour, window, speed, bullet_speed, bullet_size, bullet_colour):
        self.player_settings = {'size': size, 'colour': colour, 'speed': speed}
        self.bullet_settings = {'speed': bullet_speed, 'size': bullet_size, 'colour': bullet_colour}
        self.bullets_shot = []
        self.position = [250, 450]
        self.window = window
        # Centre of player is at self.position (as opposed to top right of player being the position).
        self.player_rect = pygame.Rect(0, 0, 2*self.player_settings['size'], 2*self.player_settings['size'])
        self.player_rect.center = self.position

    def draw(self):
        # pygame.draw.rect(self.window, self.player_settings['colour'], (
        # self.position[0] - self.player_settings['size'], self.position[1], self.player_settings['size'] * 3,
        # self.player_settings['size'] * 3))
        pygame.draw.rect(self.window, self.player_settings['colour'], self.player_rect)


    def move(self, action):
        if action[0] == 1:
            if self.player_settings['speed'] + self.player_settings['size'] <= self.position[0]:
                self.position[0] -= self.player_settings['speed']
        if action[1] == 1:
            if self.position[0] <= 500 - self.player_settings['speed'] - (self.player_settings['size'] * 2):
                self.position[0] += self.player_settings['speed']

        # if action[2] == 1:
        #     if len(self.bullets_shot) < 1:
        #         self.bullets_shot.append(Bullets(self.bullet_settings, self.position.copy(), self.window))
        # Do nothing
        if action[2] == 1:
            return
        self.player_rect.center = self.position

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
        self.is_danger = False
        self.rock_rect = pygame.Rect(0, 0, 2 * self.size, 2 * self.size)
        self.rock_rect.center = self.position

    def move(self):
        self.position[1] += self.speed
        self.rock_rect.center = self.position

    def draw(self):
        if self.is_danger:
            # pygame.draw.rect(self.window, (255, 182, 193), self.rock_rect)
            pygame.draw.circle(self.window, (255, 0, 0), self.position, self.size)
        else:
            # pygame.draw.rect(self.window, (255, 182, 193), self.rock_rect)
            pygame.draw.circle(self.window, self.colour, self.position, self.size)



class Game(object):
    def __init__(self, show_visuals=True, background_colour=(0, 0, 0), rock_colour=(255, 255, 255)):
        self.rock_colour = rock_colour
        self.background_colour = background_colour
        self.show_visuals = show_visuals
        self.window_width = 500
        self.window_height = 500
        self.rock_size_lwr_bnd = 5
        self.rock_size_upr_bnd = 20

        self.reset()

    def reset(self):
        if self.show_visuals:
            self.clock = pygame.time.Clock()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        else:
            self.window = None
        self.player = Player(20, (0, 255, 0), self.window, 10, 10 * 1.5, 5, (125, 0, 125))
        self.rocks = []
        self.rock_speed = 1
        self.speed_lwr_bnd = 1
        self.speed_upr_bnd = self.speed_lwr_bnd + 4
        self.rock_radius = 1
        self.speed_timer = 0
        self.frame_iteration = 0
        self.rocks_survived = 0
        self.score = 0

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
        size = np.random.randint(self.rock_size_lwr_bnd, self.rock_size_upr_bnd)
        position = [np.random.randint(0, 500), size]
        self.rock_speed = np.random.randint(self.speed_lwr_bnd, self.speed_upr_bnd)
        self.rock_radius = size * 2
        rock = Rock(size, self.rock_colour, self.rock_speed, position, self.window)
        self.rocks.append(rock)

    def rock_movement(self):
        self.rocks_survived = 0
        for rock in self.rocks:
            rock.move()
            if rock.position[1] - rock.size >= 500:
                self.rocks.pop(self.rocks.index(rock))
                self.rocks_survived += 1

    def danger_rocks(self, num_rocks):
        nearby_rocks = {'positions': np.zeros((num_rocks, 2)), 'speeds': np.zeros(num_rocks), 'sizes': np.zeros(num_rocks)}
        furthest_rock_idx = 0
        furthest_rock_dist = 0
        danger_rocks = [] # Stores the danger rocks which need to have their colours changed.
        for i, rock in enumerate(self.rocks):
            rock.is_danger = False
            if i < num_rocks:
                nearby_rocks['positions'][i] = rock.position
                nearby_rocks['speeds'][i] = rock.speed
                nearby_rocks['sizes'][i] = rock.size
                danger_rocks.append(rock)
                if self.compute_dist_to_rock(rock.position, rock.size) > furthest_rock_dist:
                    furthest_rock_idx = i
                    furthest_rock_dist = self.compute_dist_to_rock(rock.position, rock.size)
            elif self.compute_dist_to_rock(rock.position, rock.size) < furthest_rock_dist:
                danger_rocks[furthest_rock_idx] = rock
                nearby_rocks['positions'][furthest_rock_idx] = rock.position
                nearby_rocks['speeds'][furthest_rock_idx] = rock.speed
                nearby_rocks['sizes'][furthest_rock_idx] = rock.size
                furthest_rock_dist, furthest_rock_idx = self.find_furthest_rock(nearby_rocks, num_rocks)

        for rock in danger_rocks:
            rock.is_danger = True

        return nearby_rocks

    def compute_dist_to_rock(self, rock_pos, rock_rad):
        dist = math.sqrt(((self.player.position[0] - rock_pos[0]) ** 2) + ((self.player.position[1] - rock_pos[1]) ** 2))
        return dist - rock_rad

    def find_furthest_rock(self, nearby_rocks, num_rocks):
        furthest_rock_idx = 0
        furthest_rock_dist = 0
        for i in range(num_rocks):
            rock_pos = nearby_rocks['positions'][i]
            rock_rad = nearby_rocks['sizes'][i]
            dist = self.compute_dist_to_rock(rock_pos, rock_rad)
            if dist > furthest_rock_dist:
                furthest_rock_dist = dist
                furthest_rock_idx = i
        return furthest_rock_dist, furthest_rock_idx

    def check_bullet_collision(self):
        for bullet in self.player.bullets_shot:
            for rock in self.rocks:
                x = bullet.position[0] - rock.position[0]
                y = bullet.position[1] - rock.position[1]
                c = math.sqrt(x ** 2 + y ** 2)
                if c <= bullet.bullet_settings['size'] + rock.size:
                    self.player.bullets_shot.pop(self.player.bullets_shot.index(bullet))
                    self.rocks.pop(self.rocks.index(rock))
                    return

    def check_player_collision(self):
        for rock in self.rocks:
            collision = rock.rock_rect.colliderect(self.player.player_rect)
            if collision:
                return collision
        return False

    def play_step(self, action):
        self.frame_iteration += 1
        if self.show_visuals:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self.player.move(action)
        self.rock_movement()
        self.rock_timer()
        self.player.bullet_movement()
        self.check_bullet_collision()

        reward = 0
        game_over = False
        if self.check_player_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        reward += self.rocks_survived
        self.score += self.rocks_survived

        if self.show_visuals:
            self.update_screen()
            self.clock.tick(27)
        if self.frame_iteration / 100 > self.speed_timer:
            # print(f"Speed increase!")
            self.speed_timer += 1
            self.speed_lwr_bnd += 0.1
            self.speed_upr_bnd += 0.1

        return reward, game_over, self.score

