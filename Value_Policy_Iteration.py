import gym
import numpy as np

def valueIteration(env, gamma = 1.0):
    '''
    @in  : env - Gym Environment
           gamma - Discount Factor 
    
    @out : v[s] - Value Function
    '''
    
    v = np.zeros(env.env.nS)
    
    max_iter = 100
    eps = 1e-20
    for i in range(max_iter):
        v_prev = np.copy(v)
        for s in range(env.env.nS): 
            # iterating over all states
            q_sa = np.zeros(env.env.nA)
            for a in range(env.env.nA):
                #iterating over all actions
                for p, s_next, r, _ in env.env.P[s][a]:
                    q_sa[a] += p*(r + gamma * v_prev[s_next])
          
            v[s] = max(q_sa)
        if(np.sum(np.abs(v-v_prev))<eps):
            print('Value iteration converged at iteration %d' %(i+1))
            break    
        
    return v

def policyEvaluation(env, policy, gamma = 1.0):
    '''
    
    @in : env - environment
          policy - array of shape s*a
          containing probability of each state action pair.
    @out : v - v[s] value function on executing given policy

    '''
    v = np.zeros(env.env.nS, dtype=np.double)
    
    max_iter = 1000
    eps = 1e-20
    for i in range(max_iter):
        v_prev = np.copy(v)
        for s in range(env.env.nS): 
            # iterating over all states
            val = 0
            for a, a_prob in enumerate(policy[s]):
                #iterating over all actions
                for p, s_next, r, _ in env.env.P[s][a]:
                    val += a_prob*p*(r + gamma * v_prev[s_next])
          
            v[s] = val
        
        if(np.sum(np.abs(v-v_prev))<eps):
            print('Policy Evaluation converged at iteration %d' %(i+1))
            break    
        
    return v

def policySelection(env, v, gamma = 1.0):
    '''
    @in : env - gym environment
          v - value function
    @out : policy - s*a
    '''
    policy = np.zeros((env.env.nS,
                        env.env.nA))

    for s in range(env.env.nS): 
        # iterating over all states
        if v[s] == 0:
            continue
        #reward for each action
        r_a = np.zeros(env.env.nA) 
        for a in range(env.env.nA):
            #iterating over all actions
            reward = 0
            for p, s_next, r, _ in env.env.P[s][a]:
                reward += p*(r + gamma * v[s_next])
            r_a[a] = reward
        #Find max reward index 
        idx = np.flatnonzero(r_a == np.max(r_a))
        #Assign all max index with equal probability
        policy[s][idx] = 1/idx.shape[0]    
         
    return policy

def policyIteration(env, policy, gamma):
    '''
    @brief : runs policy evaluation and policy selection
             iteratively to converge to optimal policy.
    
    @in : random / initial policy
    @out : optimal policy and optimal value function
    '''
    
    max_iter = 1000
    eps = 1e-20
    value = np.zeros(env.env.nS)
    for i in range(max_iter):
        value_prev = np.copy(value) 
        value = policyEvaluation(env, policy, gamma)
        policy = policySelection(env, value, gamma)

        if(np.sum(np.abs(value_prev - value)) < eps):
            print('Policy Iteration converged @ iteration %d' %(i))
            break
        
    return value, policy
    
if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env.reset()
    
    gamma = 1
    
    # Value Iteration
    optimal_value = valueIteration(env, gamma)
    print("optimal value (Value Iteration): \n", optimal_value.reshape((4,4)))

    policy = policySelection(env, optimal_value, gamma)
    print ('optimal policy (Value Iteration)\n', policy)

    value = policyEvaluation(env, policy, gamma)
    print('Value from executing optimal policy \n', value.reshape(4,4))

    
    #policy iteration
    env.reset()
    policy = np.ones((env.env.nS, env.env.nA)) * 0.25
    value, policy = policyIteration(env, policy, gamma)
    print('value from policy Iteration \n', value.reshape((4,4)))
    print('policy from policy Iteration \n', policy)
    