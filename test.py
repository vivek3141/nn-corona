from env import CoronaEnv
import cv2
import neat
import pickle

im1 = cv2.imread("home.png")
im2 = cv2.imread("away.png")

state = 0
reward = 0

env = CoronaEnv()

genome = pickle.load(open("winner.pkl", 'rb'))

net = neat.nn.FeedForwardNetwork.create(genome, config)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)


i = 1

while True:
    cv2.imshow("Corona", im2 if state else im1)
    state, delta, done = env.step(net.activate(state))
    reward += delta
    print(f"Step {i}: Reward {reward} Done: {done}")
    i += 1 
    if cv2.waitKey(1000) & 0xFF == 27:
        exit()
