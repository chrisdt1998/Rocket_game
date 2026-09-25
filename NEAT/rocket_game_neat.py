"""
This file represents the rocket game which has been edited such that it can take in actions from the DQN and output the
resulting state and reward.

This file was created and designed by Christopher du Toit.
"""
import os

import pygame
import numpy as np
import math
import neat

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
        pygame.draw.rect(self.window, self.player_settings['colour'], self.player_rect)

    def move(self, action):
        if action == 0:
            if self.player_settings['speed'] + self.player_settings['size'] <= self.position[0]:
                self.position[0] -= self.player_settings['speed']
        if action == 1:
            if self.position[0] <= 500 - self.player_settings['speed'] - (self.player_settings['size']):
                self.position[0] += self.player_settings['speed']
        # Shoot bullets
        # if action[2] == 1:
        #     if len(self.bullets_shot) < 1:
        #         self.bullets_shot.append(Bullets(self.bullet_settings, self.position.copy(), self.window))
        # Do nothing
        if action == 2:
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
            pygame.draw.circle(self.window, (255, 0, 0), self.position, self.size)
        else:
            pygame.draw.circle(self.window, self.colour, self.position, self.size)



class Game(object):
    def __init__(self, show_visuals=True, background_colour=(0, 0, 0), rock_colour=(255, 255, 255), normalize_state=True):
        self.rock_colour = rock_colour
        self.background_colour = background_colour
        self.show_visuals = show_visuals
        self.normalize_state = normalize_state
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
        for player in self.players:
            player.draw()
        for rock in self.rocks:
            rock.draw()
        # for bullet in self.player.bullets_shot:
        #     bullet.draw()
        pygame.display.update()

    def rock_timer(self):
        if self.rock_radius > 0:
            self.rock_radius -= self.rock_speed
        else:
            self.rock_spawn()

    def rock_spawn(self):
        size = np.random.randint(self.rock_size_lwr_bnd, self.rock_size_upr_bnd)
        position = [np.random.randint(0 - size, 500 + size), size]
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

    def danger_rocks(self, num_rocks, player):
        nearby_rocks = {'positions': np.zeros((num_rocks, 2)), 'speeds': np.zeros(num_rocks), 'sizes': np.zeros(num_rocks)}
        furthest_rock_idx = 0
        furthest_rock_dist = 0
        danger_rocks = [] # Stores the danger rocks which need to have their colours changed.
        for i, rock in enumerate(self.rocks):
            # rock.is_danger = False
            if i < num_rocks:
                nearby_rocks['positions'][i] = rock.position
                nearby_rocks['speeds'][i] = rock.speed
                nearby_rocks['sizes'][i] = rock.size
                danger_rocks.append(rock)
                if self.compute_dist_to_rock(rock.position, rock.size, player) > furthest_rock_dist:
                    furthest_rock_idx = i
                    furthest_rock_dist = self.compute_dist_to_rock(rock.position, rock.size, player)
            elif self.compute_dist_to_rock(rock.position, rock.size, player) < furthest_rock_dist:
                danger_rocks[furthest_rock_idx] = rock
                nearby_rocks['positions'][furthest_rock_idx] = rock.position
                nearby_rocks['speeds'][furthest_rock_idx] = rock.speed
                nearby_rocks['sizes'][furthest_rock_idx] = rock.size
                furthest_rock_dist, furthest_rock_idx = self.find_furthest_rock(nearby_rocks, num_rocks, player)

        for rock in danger_rocks:
            rock.is_danger = True

        return self.reorder_danger_rocks(nearby_rocks, player)

    def compute_dist_to_rock(self, rock_pos, rock_rad, player):
        dist = math.sqrt(((player.position[0] - rock_pos[0]) ** 2) + ((player.position[1] - rock_pos[1]) ** 2))
        return dist - rock_rad

    def find_furthest_rock(self, nearby_rocks, num_rocks, player):
        furthest_rock_idx = 0
        furthest_rock_dist = 0
        for i in range(num_rocks):
            rock_pos = nearby_rocks['positions'][i]
            rock_rad = nearby_rocks['sizes'][i]
            dist = self.compute_dist_to_rock(rock_pos, rock_rad, player)
            if dist > furthest_rock_dist:
                furthest_rock_dist = dist
                furthest_rock_idx = i
        return furthest_rock_dist, furthest_rock_idx

    def reorder_danger_rocks(self, danger_rocks, player):
        dist_arr = []
        for i, rock in enumerate(danger_rocks['positions']):
            dist = self.compute_dist_to_rock(rock, danger_rocks['sizes'][i], player)
            dist_arr.append(dist)
        dist_arr = np.array(dist_arr)
        arr_order = dist_arr.argsort()
        danger_rocks['positions'] = danger_rocks['positions'][arr_order]
        danger_rocks['sizes'] = danger_rocks['sizes'][arr_order]
        danger_rocks['speeds'] = danger_rocks['speeds'][arr_order]
        return danger_rocks

    def check_bullet_collision(self, player):
        for bullet in player.bullets_shot:
            for rock in self.rocks:
                x = bullet.position[0] - rock.position[0]
                y = bullet.position[1] - rock.position[1]
                c = math.sqrt(x ** 2 + y ** 2)
                if c <= bullet.bullet_settings['size'] + rock.size:
                    player.bullets_shot.pop(player.bullets_shot.index(bullet))
                    self.rocks.pop(self.rocks.index(rock))
                    return

    def check_player_collision(self, player):
        for rock in self.rocks:
            collision = rock.rock_rect.colliderect(player.player_rect)
            if collision:
                return collision
        return False

    def get_state(self, player):
        # State contains the position of the player, nearby rocks positions, nearby rocks sizes, nearby rocks speeds.
        # The nearby rocks have been rearrange from nearest to furthest.
        nearby_rocks = self.danger_rocks(5, player)
        nearby_rock_positions = []
        nearby_rock_sizes = []
        nearby_rock_speeds = []
        if self.normalize_state:
            for pos, size, speed in zip(nearby_rocks['positions'], nearby_rocks['sizes'], nearby_rocks['speeds']):
                nearby_rock_positions += (pos/self.window_width).tolist()
                nearby_rock_sizes.append(size/self.rock_size_upr_bnd)
                nearby_rock_speeds.append(round(speed/self.speed_upr_bnd, 2))
            state = [player.position[0]/self.window_height, player.position[1]/self.window_width] + nearby_rock_positions + nearby_rock_sizes + nearby_rock_speeds
        else:
            for pos, size, speed in zip(nearby_rocks['positions'], nearby_rocks['sizes'], nearby_rocks['speeds']):
                nearby_rock_positions += pos.tolist()
                nearby_rock_sizes.append(size)
                nearby_rock_speeds.append(speed)
            state = player.position + nearby_rock_positions + nearby_rock_sizes + nearby_rock_speeds

        # print(state)
        return state

    def play_step(self):
        self.frame_iteration += 1
        if self.show_visuals:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        for player in self.players:
            idx = self.players.index(player)
            self.ge[idx].fitness += 0.1
            state = self.get_state(player)
            output = self.nets[idx].activate(state)
            action = np.argmax(np.array(output))
            player.move(action)

            if self.check_player_collision(player):
                self.ge[idx].fitness -= 1
                self.nets.pop(idx)
                self.ge.pop(idx)
                self.players.pop(idx)
        self.rock_movement()
        self.rock_timer()
        # player.bullet_movement()
        # self.check_bullet_collision()

        for genome in self.ge:
            genome.fitness += self.rocks_survived
        self.score += self.rocks_survived

        if self.show_visuals:
            self.update_screen()
            self.clock.tick(100)
        if self.frame_iteration / 100 > self.speed_timer:
            self.speed_timer += 1
            self.speed_lwr_bnd += 0.1
            self.speed_upr_bnd += 0.1
        for rock in self.rocks:
            rock.is_danger = False

    def eval_genomes(self, genomes, config):
        self.nets = []
        self.players = []
        self.ge = []
        for genome_id, genome in genomes:
            genome.fitness = 0  # start with fitness level of 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.nets.append(net)
            self.players.append(Player(20, (0, 255, 0), self.window, 10, 10 * 1.5, 5, (125, 0, 125)))
            self.ge.append(genome)
        running = True

        while running and len(self.players) > 0:
            self.play_step()

        self.reset()



def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    game = Game(show_visuals=False, normalize_state=False)
    winner = p.run(game.eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join('config-feedforward.txt')
    run(config_path)