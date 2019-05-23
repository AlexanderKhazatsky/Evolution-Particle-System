from World import World
import random


def init_size_experiment(init_plants):
    world = World(init_plants)
    for i in range(1000): world.env_step()


def mutation_experiment(mut_var):
    world = World(random.randint(3, 6), mutation_var=mut_var)
    for i in range(1000): world.env_step()


def quant_qual_experiment(specialty):
    if specialty == "quality":
        world = World(random.randint(1, 5), quantity=False)
    else:
        world = World(random.randint(1, 5))
    for i in range(1000): world.env_step()


def disaster_experiment(frequency):
    world = World(random.randint(1, 5), disaster=True, disaster_freq=frequency)
    for i in range(1000): world.env_step()


def dimension_experiment(dim):
    world = World(random.randint(1, 5), dim=(dim, dim))
    for i in range(1000): world.env_step()


def basic_demonstration():
    world = World(random.randint(2,5))
    for i in range(1000): world.env_step()

basic_demonstration() #Nothing
mutation_experiment(0) #mutation variance
quant_qual_experiment("quality") #reproduction type "quality"/"quantity"
mutation_experiment() #mutation variance
disaster_experiment(15) #disaster frequency
basic_demonstration() #Nothing
init_size_experiment() #init plants)
dimension_experiment() #world dimension (x,y)