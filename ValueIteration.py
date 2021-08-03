import numpy as np
import time
import gym

def valueIteration(env, gamma):
    value = np.zeros(env.env.nS)
    max_iter = 10000
    eps = 1e-20

    for i in range(max_iter):
        prev_v = np.copy(value)
        for s in range(env.env.nS):
            q_sa = [sum([p * (r + prev_v[s_]) 
                    for p, s_, r, _ in env.env.P[s][a]]) 
                    for a in range(env.env.nA)]
            value[s] = max(q_sa)

        if np.sum(np.fabs(prev_v - value)) <= eps:            
            print('Value iteration converged a $%d' %(i+1))
            break

    return value

def calculatePolicy(v, gamma = 1.0):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                p, s_, r, _ = next_sr
                q_sa[a] += sum( p* (r + gamma * v[s_]))
        
        policy[s] = np.argmax(q_sa)
    return policy

def evaluatePolicy()

if __name__ == '__main__':
    print('Lets Get Started')
    env = gym.make('FrozenLake-v0')
    env.reset()

    gamma = 1.0

    #optimalValue = valueIteration(env, gamma)

    print(env.env.nS)

    env.close()