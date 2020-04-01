from env import CoronaEnv
import cv2
import neat

im1 = cv2.imread("home.png")
im2 = cv2.imread("away.png")

state = 0
reward = 0

env = CoronaEnv()

net = neat.nn.FeedForwardNetwork.create(genome, config)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
genome = pickle.load(open("winner.pkl", 'rb'))


while True:
    cv2.imshow("Corona", im2 if state else im1)
    state, delta, done = env.step(net.activate(state))
    if cv2.waitKey(1000) & 0xFF == 27:
        exit()
