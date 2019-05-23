import numpy as np
from Generator import WorldGenerator
from scipy.misc import toimage
import random
from matplotlib import pyplot as plt
from matplotlib import animation

plant_colors = np.array(
    [[104/256, 230/256, 158/256],
    [114/256, 134/256, 38/256],
    [70/256, 126/256, 4/256],
    [42/256, 70/256, 8/256],
    [142/256, 170/256, 72/256]])


class World:
    def __init__(self,
                 num_plants,
                 dim=(200, 200),
                 traits_dim=5,
                 mutation_var=0.01,
                 quantity=True,
                 disaster=False,
                 disaster_freq=10
                 ):

        self.num_plants = num_plants
        self.dim = dim
        self.quantity = quantity
        self.disaster = disaster
        self.disaster_freq = disaster_freq
        self.traits_dim = traits_dim # length of our tree and soil traits vector
        self.mutation_var = mutation_var
        world = WorldGenerator(dim)
        self.trait_world, self.color_world, self.ideal_world = world.trait_world, world.color_world, world.ideal_world
        self.template_world = self.color_world.copy()
        self.world = self.initialize_world()
        self.timestep = 0
        self.initialize_plants()
        self.fitness_progress = np.zeros((1000,))
        self.spread_progress = np.zeros((1000,))
        self.convergence_progress = np.zeros((1000,))
        toimage(self.ideal_world).show()
        plt.ion()

    def initialize_world(self):
        return [[Soil((i, j), self.env_type(self.trait_world[i][j]), self.mutation_var, self.traits_dim)
                 for j in range(self.dim[1])]
                for i in range(self.dim[0])]

    def initiate_disaster(self):
        world = WorldGenerator(self.dim)
        self.trait_world, self.color_world, self.ideal_world = world.trait_world, world.color_world, world.ideal_world
        self.template_world = self.color_world.copy()
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.world[i][j].env_type = self.env_type(self.trait_world[i][j])
                self.world[i][j].is_ocean = self.world[i][j].env_type < 0

    def env_type(self, num):
        if num < 0.05:
            return -1 #ocean
        if num < 0.1:
            return 0 #shallow
        elif num < 0.15:
            return 1 #beach
        elif num < 0.2:
            return 2 #dirt
        elif num < 0.33:
            return 3 #land
        elif num < 0.45:
            return 4 #mountain
        else:
            return 5 #snow

    def plant_color(self, traits):
        return np.matmul(traits, plant_colors)

    def ideal_color(self, env_type):
        if env_type > 5:
            return [255/256, 250/256, 250/256]
        return plant_colors[env_type]

    def initialize_plants(self):
        def get_random_land_coord():
            x, y = random.randint(0, self.dim[0] - 1), random.randint(0, self.dim[1] - 1)
            while self.world[x][y].is_ocean:
                x, y = random.randint(0, self.dim[0] - 1), random.randint(0, self.dim[1] - 1)
            return x, y

        for i in range(self.num_plants):
            x, y = get_random_land_coord()
            env_initialization = np.abs(np.random.normal(0, 1, self.traits_dim))
            seed_trait = env_initialization / np.sum(env_initialization)
            seed = (seed_trait, self.timestep)
            self.world[x][y].add_seed(seed) # Now each seed is (seed_traits, timestep)

    def env_step(self):
        total_fitness = 0
        total_convergence = 0
        num_plants = 0
        self.timestep += 1
        self.animate_world()
        if self.disaster and self.timestep > 0 and self.timestep % self.disaster_freq == 0:
            self.initiate_disaster()
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.world[i][j].env_step(self.timestep)
                curr_tree = self.world[i][j].tree
                if curr_tree is not None:
                    num_plants += 1
                    total_fitness += self.world[i][j].tree_fitness(curr_tree)
                    total_convergence += self.world[i][j].convergence()
                    self.spread_seed(i, j)
                    self.color_world[i][j] = self.plant_color(curr_tree[0])
                else:
                    self.color_world[i][j] = self.template_world[i][j]

                if self.world[i][j].is_ocean:
                    seeds = self.world[i][j].seeds
                    self.world[i][j].seeds = []
                    for seed in seeds:
                        if np.random.uniform() < min(1, seed[0][0] * 3):
                            self.move_seed(i, j, seed)
        self.fitness_progress[self.timestep - 1] = round(total_fitness/num_plants, 4) if num_plants > 0 else 0
        self.spread_progress[self.timestep - 1] = num_plants
        self.convergence_progress[self.timestep - 1] = round(total_convergence / num_plants, 4) if num_plants > 0 else 0
        if (self.timestep % 10 == 0 and num_plants > 0): print("Fitness: " + str(round(total_fitness/num_plants, 3)))
        if (self.timestep % 10 == 0 and num_plants > 0): print("Convergence: " + str(round(total_convergence / num_plants, 3)))

    def animate_world(self):
        imgplot = plt.imshow(self.color_world)
        plt.show()
        plt.pause(0.01)

    def mutate_traits(self, traits, env_type):
        if self.mutation_var == 0:
            return traits
        if not self.quantity:
            mutated_traits = traits.copy()
            mutated_traits[min(env_type, 4)] += np.abs(np.random.normal(0, self.mutation_var))
            return mutated_traits / np.sum(mutated_traits)
        mutated_traits = traits + np.abs(np.random.normal(0, self.mutation_var, self.traits_dim))
        return mutated_traits / np.sum(mutated_traits)

    def spread_seed(self, i, j):
        parent_tree = self.world[i][j].tree
        survival_prob = self.world[i][j].tree_fitness(parent_tree)
        if self.quantity:
            spread_count = 1 + 3 * np.random.geometric(survival_prob)
        else:
            spread_count = np.random.geometric(survival_prob)

        for k in range(spread_count):
            x_index = int(np.round(np.clip(random.gauss(i, 2), 0, self.dim[0] - 1)))
            y_index = int(np.round(np.clip(random.gauss(j, 2), 0, self.dim[1] - 1)))
            mutated_traits = self.mutate_traits(parent_tree[0], self.world[i][j].env_type)
            seed = (mutated_traits, self.timestep)
            self.world[x_index][y_index].add_seed(seed)

    def move_seed(self, i, j, seed):
        x = int(np.round(np.clip(random.gauss(i, 2), 0, self.dim[0] - 1)))
        y = int(np.round(np.clip(random.gauss(j, 2), 0, self.dim[1] - 1)))
        self.world[x][y].add_seed(seed)

    def __repr__(self):
        repr_str = "timestep: " + str(self.timestep) + "\n"
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                repr_str += str(self.world[i][j]) + "\n"
        return repr_str


class Soil:
    def __init__(self, coord, env_type, mutation_var, traits_dim):
        self.coord = coord
        self.traits_dim = traits_dim
        self.env_type = env_type
        self.mutation_var = mutation_var
        self.tree = None
        self.seeds = []
        self.is_ocean = self.env_type < 0

    def env_step(self, timestep):
        self.age_tree()
        if not self.is_ocean:
            self.fitness_competition(timestep)

    def age_tree(self):
        if self.tree is None:
            return
        prob = self.tree_fitness(self.tree)
        if np.random.uniform() > prob:
            self.tree = None

    def fitness_competition(self, timestep):
        competitors, next_timestep_seeds = [], []
        for seed in self.seeds:
            if seed[1] == timestep - 1: # Only make seeds planted in the previous timestep compete
                competitors.append(seed) # seed[0] are the traits of the seed
            else:
                next_timestep_seeds.append(seed) # consider the seed for the next timestep
        self.seeds = next_timestep_seeds
        if not competitors:
            return
        if self.tree is not None:
            competitors += [self.tree]

        fitness = np.array([self.tree_fitness(competitors[i]) for i in range(len(competitors))])
        highest = fitness.max()
        probabilities = np.exp(1000 * (fitness - highest)) / np.sum(np.exp(1000 * (fitness - highest)))
        winner = random.choices([i for i in range(len(competitors))], probabilities)[0]
        self.tree = competitors[winner]

    def add_seed(self, seed):
        self.seeds.append(seed)

    def tree_fitness(self, tree):
        if self.env_type == 5:
            fitness = tree[0][self.env_type - 1] / 5
        else:
            fitness = tree[0][self.env_type]
        return fitness

    def convergence(self):
        assert self.tree is not None
        preferred_loc = np.argmax(self.tree[0])
        return preferred_loc == self.env_type

    def __repr__(self):
        return ("[Soil: " + str(self.coord) + ", is_water: " + str(int(self.is_ocean)) +
                ", traits: " + str(self.env_type) +
                ", tree: " + str(self.tree) + ", seeds: " + str(self.seeds) + "]")