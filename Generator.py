import noise
import math
import numpy as np
import random
from noise.perlin import SimplexNoise
from opensimplex import OpenSimplex
import math

scale = 50
octaves = 6
persistence = 0.5
lacunarity = 2.0

plant_colors = np.array(
    [[104, 230, 158],
     [114, 134, 38],
     [70, 126, 4],
     [42, 70, 8],
     [142, 170, 72]])


class WorldGenerator:
    def __init__(self, dim):
        self.dim = dim
        self.gen1 = OpenSimplex(seed=random.randint(50, 1200))
        self.gen2 = OpenSimplex(seed=random.randint(50, 1030))
        self.noise1 = lambda nx, ny: self.gen1.noise2d(nx, ny) / 2 + 0.5
        self.noise2 = lambda nx, ny: self.gen2.noise2d(nx, ny) / 2 + 0.5
        self.trait_world = self.create_trait_world(dim)
        self.color_world = self.create_colored_world(self.trait_world)
        self.ideal_world = self.create_ideal_world(self.trait_world)

    def additive_noise(self, rand, n):
        if rand < 0.15:
            return n
        elif rand < 0.3:
            return - n
        elif rand < 0.45:
            return n / 2
        elif rand < 0.6:
            return - n / 2
        elif rand < 0.75:
            return n * 2
        elif rand < 0.95:
            return - n * 2
        else:
            return 0

    def create_trait_world(self, dim):
        world = np.zeros(dim)
        rand = np.random.uniform()
        for x in range(dim[0]):
            for y in range(dim[1]):
                nx = x / dim[0] - 0.5
                ny = y / dim[1] - 0.5

                elev = (1.00 * self.noise1(1 * nx, 1 * ny)
                        + 0.50 * self.noise1(2 * nx, 2 * ny)
                        + 0.25 * self.noise1(4 * nx, 4 * ny)
                        + 0.13 * self.noise1(8 * nx, 8 * ny)
                        + 0.06 * self.noise1(16 * nx, 16 * ny)
                        + 0.03 * self.noise1(32 * nx, 32 * ny))

                elev /= (1.00 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
                elev = math.pow(elev, 2.46)

                world[x][y] = 0.3 * elev

                elev = (1.00 * self.noise2(1 * nx, 1 * ny)
                        + 0.50 * self.noise2(2 * nx, 2 * ny)
                        + 0.25 * self.noise2(4 * nx, 4 * ny)
                        + 0.13 * self.noise2(8 * nx, 8 * ny)
                        + 0.06 * self.noise2(16 * nx, 16 * ny)
                        + 0.03 * self.noise2(32 * nx, 32 * ny))

                elev /= (1.00 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
                elev = math.pow(elev, 2.46)
                world[x][y] += 0.3 * elev

                new_noise = noise.pnoise2(
                    x / scale,
                    y / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=dim[0],
                    repeaty=dim[1],
                    base=0,
                )
                world[x][y] += self.additive_noise(rand, new_noise)
        return world

    def create_colored_world(self, world):
        ocean = [65 / 256, 105 / 256, 225 / 256]
        dirt = [160 / 256, 82 / 256, 45 / 256]
        wetland = [51 / 256, 230 / 256, 255 / 256]
        land = [139 / 256, 69 / 256, 19 / 256]
        beach = [238 / 256, 214 / 256, 175 / 256]
        mountain = [139 / 256, 137 / 256, 137 / 256]
        snow = [255 / 256, 250 / 256, 250 / 256]

        color_world = np.zeros(world.shape + (3,))
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                if world[i][j] < 0.05:
                    color_world[i][j] = ocean  # seeds
                elif world[i][j] < 0.1:
                    color_world[i][j] = wetland  # mangroves
                elif world[i][j] < 0.15:
                    color_world[i][j] = beach  # palm tree
                elif world[i][j] < 0.2:
                    color_world[i][j] = dirt  # oak tree
                elif world[i][j] < 0.33:
                    color_world[i][j] = land  # pine tree
                elif world[i][j] < 0.45:
                    color_world[i][j] = mountain  # shrub
                else:
                    color_world[i][j] = snow  # nothing
        return color_world

    def create_ideal_world(self, world):
        ideal_world = np.zeros(world.shape + (3,))
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                if world[i][j] < 0.05:
                    ideal_world[i][j] = [65, 105, 225]  # ocean
                elif world[i][j] < 0.1:
                    ideal_world[i][j] = plant_colors[0]  # mangroves
                elif world[i][j] < 0.15:
                    ideal_world[i][j] = plant_colors[1]  # palm tree
                elif world[i][j] < 0.2:
                    ideal_world[i][j] = plant_colors[2]  # oak tree
                elif world[i][j] < 0.33:
                    ideal_world[i][j] = plant_colors[3]  # pine tree
                elif world[i][j] < 0.45:
                    ideal_world[i][j] = plant_colors[4]  # scrub
                else:
                    ideal_world[i][j] = [255, 250, 250]  # snow
        return ideal_world

    def create_island_world(self, world):
        center_x, center_y = self.dim[1] // 2, self.dim[0] // 2
        circle_grad = np.zeros_like(world)

        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                distx = abs(x - center_x)
                disty = abs(y - center_y)
                dist = math.sqrt(distx * distx + disty * disty)
                circle_grad[y][x] = dist

        max_grad = np.max(circle_grad)
        circle_grad = circle_grad / max_grad
        circle_grad -= 0.5
        circle_grad *= 2.0
        circle_grad = -circle_grad

        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                if circle_grad[y][x] > 0:
                    circle_grad[y][x] *= 20

        max_grad = np.max(circle_grad)
        circle_grad = circle_grad / max_grad
        world_noise = np.zeros_like(world)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                world_noise[i][j] = (world[i][j] * circle_grad[i][j])
                if world_noise[i][j] > 0:
                    world_noise[i][j] *= 20

        max_grad = np.max(world_noise)
        world_noise = world_noise / max_grad
        world_noise += world
        return world_noise
