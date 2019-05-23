from World import World
import numpy as np
from Generator import WorldGenerator
from scipy.misc import toimage
import random
from matplotlib import pyplot as plt
from matplotlib import animation


fig_number = 0


def base_experiment(init_plants):
    global fig_number
    loc = "/Users/sasha/desktop/plant_data/base_experiment"
    world = World(init_plants)
    toimage(world.ideal_world).save(
        loc + "/{0}/ideal_world_{1}.png".format(init_plants, init_plants))
    for i in range(1000):
        if i < 100 and i % 10 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(init_plants, i, init_plants))
        if i % 100 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(init_plants, i, init_plants))
        world.env_step()
    toimage(world.color_world).save(
        loc + "/{0}/final_world_{1}.png".format(init_plants, init_plants))

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Population Growth")
    plt.xlabel("Time")
    plt.ylabel("Plants Alive")
    plt.plot(world.spread_progress)
    plt.savefig(loc + "/{0}/population_progress_{1}.png".format(init_plants, init_plants))
    np.save(loc + "/{0}/population_progress_{1}".format(init_plants, init_plants), world.spread_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Convergence Progress")
    plt.xlabel("Time")
    plt.ylabel("Percentage")
    plt.plot(world.convergence_progress)
    plt.savefig(loc + "/{0}/convergence_progress_{1}.png".format(init_plants, init_plants))
    np.save(loc + "/{0}/convergence_progress_{1}".format(init_plants, init_plants), world.convergence_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Fitness Progress")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.plot(world.fitness_progress)
    plt.savefig(loc + "/{0}/fitness_progress_{1}.png".format(init_plants, init_plants))
    np.save(loc + "/{0}/fitness_progress_{1}".format(init_plants, init_plants), world.fitness_progress)


def mutation_experiment(mut_var):
    global fig_number
    loc = "/Users/sasha/desktop/plant_data/mutation_var"
    world = World(random.randint(3, 6), mutation_var=mut_var)
    toimage(world.ideal_world).save(
        loc + "/{0}/ideal_world_{1}.png".format(mut_var, mut_var))
    for i in range(1000):
        if i < 100 and i % 10 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(mut_var, i, mut_var))
        if i % 100 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(mut_var, i, mut_var))
        world.env_step()
    toimage(world.color_world).save(
        loc + "/{0}/final_world_{1}.png".format(mut_var, mut_var))

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Population Growth")
    plt.xlabel("Time")
    plt.ylabel("Plants Alive")
    plt.plot(world.spread_progress)
    plt.savefig(loc + "/{0}/population_progress_{1}.png".format(mut_var, mut_var))
    np.save(loc + "/{0}/population_progress_{1}".format(mut_var, mut_var), world.spread_progress)


    fig_number += 1
    plt.figure(fig_number)
    plt.title("Convergence Progress")
    plt.xlabel("Time")
    plt.ylabel("Percentage")
    plt.plot(world.convergence_progress)
    plt.savefig(loc + "/{0}/convergence_progress_{1}.png".format(mut_var, mut_var))
    np.save(loc + "/{0}/convergence_progress_{1}".format(mut_var, mut_var), world.convergence_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Fitness Progress")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.plot(world.fitness_progress)
    plt.savefig(loc + "/{0}/fitness_progress_{1}.png".format(mut_var, mut_var))
    np.save(loc + "/{0}/fitness_progress_{1}".format(mut_var, mut_var), world.fitness_progress)


def quant_qual_experiment(specialty):
    global fig_number
    loc = "/Users/sasha/desktop/plant_data/quantity_quality_experiment"
    if specialty == "quality":
        world = World(random.randint(1, 5), quantity=False)
    else:
        world = World(random.randint(1, 5))

    toimage(world.ideal_world).save(
        loc + "/{0}/ideal_world_{1}.png".format(specialty, specialty))

    for i in range(1000):
        if i < 100 and i % 10 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(specialty, i, specialty))
        if i % 100 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(specialty, i, specialty))
        world.env_step()

    toimage(world.color_world).save(
        loc + "/{0}/final_world_{1}.png".format(specialty, specialty))

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Population Growth")
    plt.xlabel("Time")
    plt.ylabel("Plants Alive")
    plt.plot(world.spread_progress)
    plt.savefig(loc + "/{0}/population_progress_{1}.png".format(specialty, specialty))
    np.save(loc + "/{0}/population_progress_{1}".format(specialty, specialty), world.spread_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Convergence Progress")
    plt.xlabel("Time")
    plt.ylabel("Percentage")
    plt.plot(world.convergence_progress)
    plt.savefig(loc + "/{0}/convergence_progress_{1}.png".format(specialty, specialty))
    np.save(loc + "/{0}/convergence_progress_{1}".format(specialty, specialty), world.convergence_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Fitness Progress")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.plot(world.fitness_progress)
    plt.savefig(loc + "/{0}/fitness_progress_{1}.png".format(specialty, specialty))
    np.save(loc + "/{0}/fitness_progress_{1}".format(specialty, specialty), world.fitness_progress)


def disaster_experiment(frequency):
    global fig_number
    loc = "/Users/sasha/desktop/plant_data/disaster_experiment"

    world = World(random.randint(1, 5), disaster=True, disaster_freq=frequency)

    toimage(world.ideal_world).save(
        loc + "/{0}/ideal_world_{1}.png".format(frequency, frequency))
    for i in range(1000):
        if i < 100 and i % 10 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(frequency, i, frequency))
        if i % 100 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(frequency, i, frequency))
        world.env_step()
    toimage(world.color_world).save(
        loc + "/{0}/final_world_{1}.png".format(frequency, frequency))

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Population Growth")
    plt.xlabel("Time")
    plt.ylabel("Plants Alive")
    plt.plot(world.spread_progress)
    plt.savefig(loc + "/{0}/population_progress_{1}.png".format(frequency, frequency))
    np.save(loc + "/{0}/population_progress_{1}".format(frequency, frequency), world.spread_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Convergence Progress")
    plt.xlabel("Time")
    plt.ylabel("Percentage")
    plt.plot(world.convergence_progress)
    plt.savefig(loc + "/{0}/convergence_progress_{1}.png".format(frequency, frequency))
    np.save(loc + "/{0}/convergence_progress_{1}".format(frequency, frequency), world.convergence_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Fitness Progress")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.plot(world.fitness_progress)
    plt.savefig(loc + "/{0}/fitness_progress_{1}.png".format(frequency, frequency))
    np.save(loc + "/{0}/fitness_progress_{1}".format(frequency, frequency), world.fitness_progress)


def dimension_experiment(dim):
    global fig_number
    loc = "/Users/sasha/desktop/plant_data/dimension_experiment"
    world = World(random.randint(1, 5), dim=(dim, dim))
    toimage(world.ideal_world).save(
        loc + "/{0}/ideal_world_{1}.png".format(dim, dim))
    for i in range(1000):
        if i < 100 and i % 10 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(dim, i, dim))
        if i % 100 == 0: toimage(world.color_world).save(
            loc + "/{0}/time_{1}_world_{2}.png".format(dim, i, dim))
        world.env_step()
    toimage(world.color_world).save(
        loc + "/{0}/final_world_{1}.png".format(dim, dim))

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Population Growth")
    plt.xlabel("Time")
    plt.ylabel("Plants Alive")
    plt.plot(world.spread_progress)
    plt.savefig(loc + "/{0}/population_progress_{1}.png".format(dim, dim))
    np.save(loc + "/{0}/population_progress_{1}".format(dim, dim), world.spread_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Convergence Progress")
    plt.xlabel("Time")
    plt.ylabel("Percentage")
    plt.plot(world.convergence_progress)
    plt.savefig(loc + "/{0}/convergence_progress_{1}.png".format(dim, dim))
    np.save(loc + "/{0}/convergence_progress_{1}".format(dim, dim), world.convergence_progress)

    fig_number += 1
    plt.figure(fig_number)
    plt.title("Fitness Progress")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.plot(world.fitness_progress)
    plt.savefig(loc + "/{0}/fitness_progress_{1}.png".format(dim, dim))
    np.save(loc + "/{0}/fitness_progress_{1}".format(dim, dim), world.fitness_progress)


def main_demonstration():
    world = World(random.randint(1,5))
    for i in range(1000):
        world.env_step()


main_demonstration()


var = [0, 1, 0.5, 0.1, 0.01, 0.001]
for v in var:
    print("New experiment")
    try:
        mutation_experiment(v)
    except:
        continue

specialty = ["quality", "quantity"]
for s in specialty:
    print("New experiment")
    try:
        quant_qual_experiment(s)
    except:
        continue

freq = [50, 150, 250, 500]
for f in freq:
    print("New experiment")
    try:
        disaster_experiment(f)
    except:
        continue


born = [1, 2, 5, 10]
for b in born:
    print("New experiment")
    try:
        base_experiment(b)
    except:
        continue

dim = [25, 50, 75, 100, 125, 150, 200]
for d in dim:
    print("New experiment")
    try:
        dimension_experiment(d)
    except:
        continue
