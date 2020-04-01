import cv2


class CoronaEnv:
    def __init__(self):
        self.state = 0
        self.im1 = cv2.imread("home.png")
        self.im2 = cv2.imread("away.png")

        cv2.imshow("Corona", self.im1)

    def step(self, action):
        """
        :param action:
            0 -> stay home
            1 -> go outside
        """
        cv2.imshow("Corona", self.im2 if action else self.im1)
        reward = -1 if action else 1
        done = not(bool(action))
        return action, reward, done

    def reset(self):
        self.state = 0
        cv2.imshow("Corona", self.im1)

