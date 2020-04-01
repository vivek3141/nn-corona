import cv2
import os
import neat
import pickle
from env import CoronaEnv

im1 = cv2.imread("home.png")
im2 = cv2.imread("away.png")

#os.chdir("checkpoints")

def get_fitness(genome, config):
    env = CoronaEnv()
    state = 0
    done = False
    reward = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    while not done:
        cv2.imshow("Corona", im2 if state else im1)
        state, delta, done = env.step(net.activate([state]))
        reward += delta

        if cv2.waitKey(1000) & 0xFF == 27:
            exit()

    return reward

def eval_genomes(genomes, config):
    for idx, genome in genomes:
        genome.fitness = get_fitness(genome, config)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,"config")
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.Checkpointer(generation_interval=5))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
winner = p.run(eval_genomes, 10)
pickle.dump(winner, open('winner.pkl', 'wb'))
