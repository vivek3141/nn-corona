import os
import neat
import pickle
from env import CoronaEnv
import cv2

#os.chdir("checkpoints")

def get_fitness(genome, config):
    env = CoronaEnv()
    state = 0
    done = False
    reward = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    while not done:
        state, delta, done = env.step(net.activate([state]))
        fitness += delta

        if cv2.waitkey(1000) & 0xFF:
            exit()

    return fitness

def eval_genomes(genomes, config):
    for idx, genome in genomes:
        i.fitness = get_fitness(genome, config)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,"config")
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.Checkpointer(generation_interval=5))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
winner = p.run(eval_genomes, 10)
pickle.dump(winner, open('winner.pkl', 'wb'))
