from env import CoronaEnv
import time
import cv2

env = CoronaEnv()

while True:
    env.step(1)
    #time.sleep(1)
    if cv2.waitKey(1000) & 0xFF == 27:
        exit()
    env.step(0)
    #time.sleep(1)
    if cv2.waitKey(1000) & 0xFF == 27:
        exit()

