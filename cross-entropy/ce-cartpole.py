import gym
import numpy as np

import operator
import heapq

"""
    Implementation of cross entropy method applied to the Cart Pole
    environment, as described by [1], at exercise 2.1.1.

    References: experiment done by @devforfu [2], thank you!

    [1]: https://gym.openai.com/docs/rl
    [2]: https://gym.openai.com/evaluations/eval_PljKu4EgQCiYEKveHJQJQ
"""

class DiscretePolicyLearner:
    def __init__(self, env, elite_rate, samples):
        self.env = env
        self.elite_s = int(elite_rate*samples)
        self.samples = samples

        dim_env = env.observation_space.shape[0]+1 # dimension of observations parameters
        dim_act = env.action_space.n               # dimension of possible actions

        self.std = np.ones(dim_env*dim_act)
        self.mean = np.zeros(dim_env*dim_act)

    # generate thetas for number of samples
    def generate_t(self):
        # fit covariance matrix over a gaussian distribution
        cov = np.diag(np.array(self.std**2))

        # initialize mean and standard deviation
        theta = np.random.multivariate_normal(mean=self.mean,
                                              cov=cov,
                                              size=self.samples
                                             )

        return theta

    def sample(self, state, weights):
        dim_env = self.env.observation_space.shape[0]
        dim_act = self.env.action_space.n

        W = weights[0 : dim_env*dim_act].reshape(dim_env, dim_act)
        b = weights[dim_env*dim_act:(dim_env+1)*dim_act].reshape(1, dim_act)

        res = state.dot(W) + b
        a = res.argmax()

        return a

    def noisy_evaluation(self, weights, limit=1000):
        cur_s = env.reset()
        total_reward = 0

        for i in range(limit):
            cur_a = self.sample(cur_s, weights)

            env.render()

            cur_s, reward, done, _ = env.step(cur_a)
            total_reward += reward

            if done:
                break

        return weights, total_reward

    def learn(self, simulations):
        for i in range(simulations):
            # generate weights
            thetas = self.generate_t()

            rewards = np.array(map(self.noisy_evaluation, thetas))

            elite_theta, avg_rwd = self.elite(rewards)

            self.mean = np.mean(elite_theta, axis=0)
            self.std = np.std(elite_theta, axis=0)

            print("Average reward for "
                  "iteration %d is %d" % (i, avg_rwd))


    def elite(self, population):
        # select weights with the best found reward
        best = heapq.nlargest(self.elite_s, population, key=operator.itemgetter(1))
        weights = [w for w, r in best]
        reward = np.mean([r for w, r in best])

        return weights, reward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    total_it = 10       # total iterations
    samples = 50        # similar to batch size
    elite_rate = 0.2    # rate of elite set

    pol = DiscretePolicyLearner(env, elite_rate, samples)

    pol.learn(total_it)
