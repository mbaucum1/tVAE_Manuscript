import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np
import argparse
import os
from queue import Queue
import multiprocessing
import threading
import pandas as pd
import matplotlib.pyplot as plt

tf.enable_eager_execution()

dataX_trim=pd.read_csv("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/dataX.csv")
dataY_trim=pd.read_csv("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/dataY.csv")

dataX_trim=np.array(dataX_trim.drop("Unnamed: 0",axis=1))
dataY_trim=np.array(dataY_trim.drop("Unnamed: 0",axis=1))

#Read in means/std (these haven't changed, despite change to HMM method)
z_means=np.loadtxt("z_means.txt",delimiter=",")
z_std=np.loadtxt("z_std.txt",delimiter=",")

fulldata=pd.read_csv("Nemani Patient Data Cleaned Imputed.csv")
var_min=np.min(dataY_trim,axis=0)
var_max=np.max(dataY_trim,axis=0)


def reward_func(aPTT):
    new_aptt=aPTT*z_std[0]+z_means[0]
    reward_val=2/(1+np.exp(-(new_aptt-60)))-2/(1+np.exp(-(new_aptt-100)))-1
    return reward_val

#Create parser
parser = argparse.ArgumentParser(description='Run A3C algorithm on medication HMM environment.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
    #Need to specify random/baseline strategy later
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=10, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=100, type=int,
                    help='Global maximum number of episodes to run.')
                    #THIS IS IMPORTANT. This is how many 'rounds' of patient data
                    #get fed into the model.
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()

#Create HMM environment
hep1_trans=np.loadtxt("hep1_transmat.txt",delimiter=",")
hep2_trans=np.loadtxt("hep2_transmat.txt",delimiter=",")
hep3_trans=np.loadtxt("hep3_transmat.txt",delimiter=",")
hep4_trans=np.loadtxt("hep4_transmat.txt",delimiter=",")
hep5_trans=np.loadtxt("hep5_transmat.txt",delimiter=",")
hep6_trans=np.loadtxt("hep6_transmat.txt",delimiter=",")

hmm_startprob=np.loadtxt("hmm_start.txt",delimiter=",")
hmm_means=np.loadtxt("hmm_means.txt",delimiter=",")
hmm_vars=np.loadtxt("hmm_vars.txt",delimiter=",")

class hmm_env():
    def __init__(self):
        self.startprob=hmm_startprob
        self.means=hmm_means
        self.vars=hmm_vars
        self.trans=[hep1_trans,hep2_trans,hep3_trans,hep4_trans,hep5_trans,hep6_trans]
        self.state=0
    def reset(self):
        done=False
        while done==False:
            self.state=np.random.choice(range(9),p=self.startprob)
            meas=np.random.multivariate_normal(mean=self.means[self.state,:],cov=np.diag(self.vars[self.state,:]))
            aptt=meas[0]*z_std[0]+z_means[0]
            if (aptt<60): 
                done=True
        return meas
    
    def transition(self,action): #Action takes 0 through 4, not 1 through 5
        transmat=self.trans[action]
        trans_dist=transmat[self.state,:]
        
        prob_vec=trans_dist
        prob_vec=np.clip(prob_vec,a_min=.001,a_max=.999)
        prob_vec=prob_vec/np.sum(prob_vec)
                        
        self.state=np.random.choice(range(9),p=prob_vec)
        meas=np.random.multivariate_normal(mean=self.means[self.state,:],cov=np.diag(self.vars[self.state,:]))
        return meas

    def simulate_actions(self,steps=7*24,trials=1000):
       final_rewards=[]
      
       for trial in range(trials):      
           if trial%10==0:
              print(trial)
           reward_vec=[0,0,0,0,0,0]
          
           self.reset()
           start_state=self.state
          
           for chosen_action in [0,1,2,3,4,5]:
               r=0
               self.state=start_state
               for i in range(steps):
                   meas=self.transition(chosen_action)
                   r+=reward_func(meas[0])*(.99**i)
               reward_vec[chosen_action]=r
           final_rewards.append(reward_vec)        
       return final_rewards


#Read in VAE parameters
encoder1_weights=np.loadtxt('encoder1_weights.txt',delimiter=',')
encoder1_bias=np.loadtxt('encoder1_bias.txt',delimiter=',')
z_mean_weights=np.loadtxt('z_mean_weights.txt',delimiter=',')
z_mean_bias=np.loadtxt('z_mean_bias.txt',delimiter=',')
z_logvar_weights=np.loadtxt('z_logvar_weights.txt',delimiter=',')
z_logvar_bias=np.loadtxt('z_logvar_bias.txt',delimiter=',')
decoder1_weights=np.loadtxt('decoder1_weights.txt',delimiter=',')
decoder1_bias=np.loadtxt('decoder1_bias.txt',delimiter=',')
output_weights=np.loadtxt('output_weights.txt',delimiter=',')
output_bias=np.loadtxt('output_bias.txt',delimiter=',')


#VAE model class
class vae_env(keras.Model):
  def __init__(self):
    super(vae_env, self).__init__()
    self.encoder_hidden=layers.Dense(encoder1_weights.shape[1],activation='sigmoid',kernel_initializer=tf.constant_initializer(encoder1_weights),bias_initializer=tf.constant_initializer(encoder1_bias))
    self.z_mean=layers.Dense(z_mean_weights.shape[1],kernel_initializer=tf.constant_initializer(z_mean_weights),bias_initializer=tf.constant_initializer(z_mean_bias))
    self.z_logvar=layers.Dense(z_mean_weights.shape[1],kernel_initializer=tf.constant_initializer(z_logvar_weights),bias_initializer=tf.constant_initializer(z_logvar_bias))
    
    def sampling(args):
        z_mean, z_logvar=args
        epsilon=tf.random_normal(shape=(int(z_mean.shape[0]),z_mean_weights.shape[1]),mean=0,stddev=1)
        return z_mean+tf.math.sqrt(tf.math.exp(z_logvar))*epsilon
       
    self.latent_hidden=layers.Lambda(sampling)
    self.decoder_hidden=layers.Dense(decoder1_weights.shape[1],activation='sigmoid',kernel_initializer=tf.constant_initializer(decoder1_weights),bias_initializer=tf.constant_initializer(decoder1_bias))
    self.next_state=layers.Dense(output_weights.shape[1],kernel_initializer=tf.constant_initializer(output_weights),bias_initializer=tf.constant_initializer(output_bias))

  def predict(self,inputs):
      input_tensor=tf.convert_to_tensor(inputs,dtype=tf.float32)
      _encoder_hidden=self.encoder_hidden(input_tensor)
      _z_mean=self.z_mean(_encoder_hidden)
      _z_logvar=self.z_logvar(_encoder_hidden)
      _latent_hidden=self.latent_hidden([_z_mean,_z_logvar])
      _decoder_hidden=self.decoder_hidden(_latent_hidden)
      _next_state=self.next_state(_decoder_hidden)   
      return _next_state.numpy()

  def reset(self):
      done=False
      while done==False:
          input_vec=np.random.normal(0,1,size=(1,14))
          input_vec=np.concatenate([input_vec,np.array([1,0,0,0,0,0]).reshape(1,6)],axis=1)
          state=self.predict(input_vec)
          aptt=state[0][0]*z_std[0]+z_means[0]
          if (aptt<60): 
              done=True
      self.state=state     
      return(self.state[0])

  def transition(self,action): #Action goes from 0 to 4
      action_vec=np.array([0,0,0,0,0,0]).reshape(1,6)
      action_vec[0,action]=1
      input_vec=np.concatenate([self.state,action_vec],axis=1)
      self.state=self.predict(input_vec)
      return(self.state[0])

  def simulate_actions(self,steps=7*24,trials=1000):
      final_rewards=[]
      
      for trial in range(trials):      
          if trial%10==0:
              print(trial)
          reward_vec=[0,0,0,0,0,0]
          
          self.reset()
          start_state=self.state
                    
          for chosen_action in [0,1,2,3,4,5]:
              r=0
              self.state=start_state
              for i in range(steps):
                  self.transition(chosen_action)
                  r+=reward_func(self.state[0][0])*(.99**i) 
              reward_vec[chosen_action]=r
          final_rewards.append(reward_vec)        
      return final_rewards
     
        
#Create LSTM environment
class lstm_env:
    def __init__(self):
        self.model=tf.keras.Sequential([
            tf.keras.layers.LSTM(units=10,activation='tanh',input_shape=[47,20],return_sequences=True,use_bias=False),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(14,activation='linear',use_bias=False))                
            ])
        self.model.load_weights("lstm_weights.h5")
        self.emis_sds=np.loadtxt("lstm_sds.txt",delimiter=",")
        self.history=np.array([]).reshape([1,-1,20])
        
    def reset(self):
        done=False
        while done==False:
            input_vec=np.random.normal(0,1,size=(1,14))
            input_vec=np.concatenate([input_vec,np.array([1,0,0,0,0,0]).reshape(1,6)],axis=1)
            input_vec=input_vec.reshape([1,1,20])
            state_mean=self.model.predict(input_vec)
            state=np.random.normal(loc=state_mean,scale=self.emis_sds)
            aptt=state[0][0][0]*z_std[0]+z_means[0]
            if (aptt<60): 
                done=True
        self.state=state[0]     
        return(self.state[0])

    def transition(self,action): #Action goes from 0 to 4
        action_vec=np.array([0,0,0,0,0,0]).reshape(1,6)
        action_vec[0,action]=1
        input_vec=np.concatenate([self.state,action_vec],axis=1)
        self.history=np.concatenate([self.history,input_vec.reshape([1,1,20])],axis=1)
        self.history=self.history[:,-8:,:]
        state_mean=self.model.predict(self.history)[:,-1,:]
        self.state=np.random.normal(loc=state_mean,scale=self.emis_sds)
        self.state=np.clip(self.state,var_min,var_max)        
        return(self.state[0])

    def multi_transition(self,action): #Where 'state' is a trials*14 array
        action_vec=np.array([0,0,0,0,0,0]).reshape(1,6)
        action_vec[0,action]=1
        action_array=np.array([action_vec.reshape(-1)]*self.state.shape[0])
        input_array=np.concatenate([self.state,action_array],axis=1)
        self.history=np.concatenate([self.history,input_array.reshape([-1,1,20])],axis=1)
        self.history=self.history[:,-8:,:]
        state_mean=self.model.predict(self.history)[:,-1,:]
        self.state=np.random.normal(loc=state_mean,scale=self.emis_sds) #Distributes the sds across rows (not columns), as it should
        self.state=np.clip(self.state,var_min,var_max) 
        return(self.state)
                
    def simulate_actions(self,steps=7*24,trials=1000):
        final_rewards=[]
      
        for trial in range(trials):      
            if trial%10==0:
                print(trial)
            reward_vec=[0,0,0,0,0,0]
          
            self.reset()
            start_state=self.state
                    
            for chosen_action in [0,1,2,3,4,5]:
                r=0
                self.state=start_state
                for i in range(steps):
                    self.transition(chosen_action)
                    r+=reward_func(self.state[0][0])*(.99**i) 
                reward_vec[chosen_action]=r
            final_rewards.append(reward_vec)        
        return final_rewards

#Create GAN environment
gan_hidden_weights=np.loadtxt('gan_hidden_weights.txt',delimiter=',')
gan_hidden_bias=np.loadtxt('gan_hidden_bias.txt',delimiter=',')
gan_output_weights=np.loadtxt('gan_output_weights.txt',delimiter=',')
gan_output_bias=np.loadtxt('gan_output_bias.txt',delimiter=',')

import keras.backend as K
def add_noise(_input):
    epsilon=K.random_normal(shape=(K.shape(_input)[0],7),mean=0,stddev=1)
    return K.concatenate([_input,epsilon],axis=1)

class gan_env:
    def __init__(self):
        self.model=tf.keras.Sequential()
        self.model.add(tf.keras.layers.Lambda(add_noise))
        self.model.add(tf.keras.layers.Dense(10,activation='sigmoid',kernel_initializer=tf.constant_initializer(gan_hidden_weights),bias_initializer=tf.constant_initializer(gan_hidden_bias)))
        self.model.add(tf.keras.layers.Dense(14,activation='linear',kernel_initializer=tf.constant_initializer(gan_output_weights),bias_initializer=tf.constant_initializer(gan_output_bias)))
    def reset(self):
        done=False
        while done==False:
            input_vec=np.random.normal(0,1,size=(1,14))
            input_vec=np.concatenate([input_vec,np.array([1,0,0,0,0,0]).reshape(1,6)],axis=1)
            self.state=self.model.predict(input_vec)
            aptt=self.state[0][0]*z_std[0]+z_means[0]
            if (aptt<60): 
                done=True
        return(self.state[0])

    def transition(self,action):
      action_vec=np.array([0,0,0,0,0,0]).reshape(1,6)
      action_vec[0,action]=1
      input_vec=np.concatenate([self.state,action_vec],axis=1)
      self.state=self.model.predict(input_vec)
      return(self.state[0])

#Create CVAE environment
      #Loading in version with .25 weighting
cvae_hidden_weights=np.loadtxt('cvae_decoder1_weights_25.txt',delimiter=',')
cvae_hidden_bias=np.loadtxt('cvae_decoder1_bias_25.txt',delimiter=',')
cvae_output_weights=np.loadtxt('cvae_output_weights_25.txt',delimiter=',')
cvae_output_bias=np.loadtxt('cvae_output_bias_25.txt',delimiter=',')

import keras.backend as K
def add_noise_cvae(_input):
    epsilon=K.random_normal(shape=(K.shape(_input)[0],7),mean=0,stddev=1)
    return K.concatenate([epsilon,_input],axis=1)
   
class cvae_env():
    def __init__(self):
        self.model=tf.keras.Sequential()
        self.model.add(tf.keras.layers.Lambda(add_noise_cvae))
        self.model.add(tf.keras.layers.Dense(10,activation='sigmoid',kernel_initializer=tf.constant_initializer(cvae_hidden_weights),bias_initializer=tf.constant_initializer(cvae_hidden_bias)))
        self.model.add(tf.keras.layers.Dense(14,activation='linear',kernel_initializer=tf.constant_initializer(cvae_output_weights),bias_initializer=tf.constant_initializer(cvae_output_bias)))

    def reset(self):
        done=False
        while done==False:
            input_vec=np.random.normal(0,1,size=(1,14))
            input_vec=np.concatenate([input_vec,np.array([1,0,0,0,0,0]).reshape(1,6)],axis=1)
            self.state=self.model.predict(input_vec)
            aptt=self.state[0][0]*z_std[0]+z_means[0]
            if (aptt<60): 
                done=True
        return(self.state[0])

    def transition(self,action):
      action_vec=np.array([0,0,0,0,0,0]).reshape(1,6)
      action_vec[0,action]=1
      input_vec=np.concatenate([self.state,action_vec],axis=1)
      self.state=self.model.predict(input_vec)
      return(self.state[0])



#START FROM HERE

#Create max number of worker steps
max_worker_steps=200 #This is how many hours you want from each worker

#Specify number of actions
action_size=6
state_size=14
#Create global losses list
global_losses=[]


#Create AC model class
class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense_policy = layers.Dense(8, activation='sigmoid')
    self.output_policy = layers.Dense(action_size)
    self.dense_reward = layers.Dense(8, activation='sigmoid')
    self.output_values = layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    x = self.dense_policy(inputs)
    policy_output = self.output_policy(x)
    v1 = self.dense_reward(inputs)
    reward_output = self.output_values(v1)
    return policy_output, reward_output

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"

  )
  result_queue.put(global_ep_reward)
  return global_ep_reward

#RANDOM AGENT DEFINED HERE - SKIPPING FOR NOW

class MasterAgent:
  def __init__(self):
    self.game_name = 'heparin_dosage' 
    save_dir = args.save_dir
    print("save loc", save_dir)
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    self.state_size = state_size  
    self.action_size = action_size   
    self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
    self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
    
  def train(self,environment):
    
    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, environment=environment,game_name=self.game_name,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]#multiprocessing.cpu_count())] 
    
    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []  # record episode reward to plot
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    return Worker.episode_reward_list, moving_average_rewards


  def play(self,steps=24*7,trials=1000,discount=.99, stochastic=False,environment='hmm'):
     self.simulated_discounted_rewards=[]
     self.action_list=[]
     if environment=='hmm':
        self.env=hmm_env()
     if environment=='vae':
        self.env=vae_env()
     if environment=='lstm':
         self.env=lstm_env()
     if environment=='gan':
         self.env=gan_env()
     if environment=='cvae':
         self.env=cvae_env()
 
     for trial in range(trials):
        print('trial '+str(trial))
        state=self.env.reset()
        
        model = self.global_model
        #May need to re-do so that it loads in its own best model, rather than the most recent
        #But using a \tmp\ directory means HMM weights wrote over the Vae weights
        model.set_weights(self.global_model.get_weights())
        done = False
        reward_sum = 0

        for step in range(steps):
        #env.render(mode='rgb_array')
            policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            policy = tf.nn.softmax(policy)
            if stochastic==False:
                action = np.argmax(policy)
            if stochastic==True:
                prob_vec=policy.numpy()[0]
                prob_vec=np.clip(prob_vec,a_min=.001,a_max=.999)
                prob_vec=prob_vec/np.sum(prob_vec)
                action = np.random.choice(self.action_size, p=prob_vec)
            self.action_list.append(action)
            state=self.env.transition(action=action)
            reward_val=reward_func(state[0])
            reward_sum += reward_val*(discount**step)
            if step in [0,50,100,150,200,250,300,350,400]:
            	print("{}. Reward: {}, action: {}".format(step, reward_sum, action))
            
            if step==steps:
                done=True
        
        self.simulated_discounted_rewards.append(reward_sum)

  def play_untreated(self,steps=24*7,trials=1000,discount=.99, environment='hmm'):
     self.simulated_discounted_rewards=[]
     if environment=='hmm':
        self.env=hmm_env()
     if environment=='vae':
        self.env=vae_env()
     if environment=='lstm':
         self.env=lstm_env()
     if environment=='gan':
         self.env=gan_env()
     if environment=='cvae':
         self.env=cvae_env()
 
     if environment!='lstm':
         for trial in range(trials):
            if trial%10==0:
                print('trial '+str(trial))
            self.env.reset()
            reward_sum = 0
    
            for step in range(steps):
            #env.render(mode='rgb_array')
                action=0
                state=self.env.transition(action=action)
                reward_val=reward_func(state[0])
                reward_sum += reward_val*(discount**step)
                if (step in [150]) and (trial%10==0):
                	print("{}. Reward: {}, action: {}".format(step, reward_sum, action))
                                        
            self.simulated_discounted_rewards.append(reward_sum)
     if environment=='lstm':
         for trial in range(trials):
            print(trial)
            self.env.reset()
            self.env.state=np.repeat(self.env.state,100,axis=0)
            self.env.history=np.array([]).reshape([100,0,20])
            reward_sum=np.zeros(100)
            for step in range(steps):
                action=0
                state=self.env.multi_transition(action=action)
                reward_val=np.array([reward_func(state[i,0]) for i in range(100)])               
                reward_sum=reward_sum+reward_val*(discount**step)

            self.simulated_discounted_rewards.append(reward_sum)
         self.simulated_discounted_rewards=np.array(self.simulated_discounted_rewards).reshape(-1)

class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []

class Worker(threading.Thread):
  # Set up global variables across different threads
  global_episode = 0
  # Moving average reward
  global_moving_average_reward = 0
  episode_reward_list=[]
  best_score = 0
  save_lock = threading.Lock()
  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,environment,
               game_name='heparin_dosage',
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCriticModel(self.state_size, self.action_size)
    self.worker_idx = idx
    self.game_name = game_name
    self.save_dir = save_dir
    self.ep_loss = 0.0
    if environment=='hmm':
        self.env=hmm_env()
    if environment=='vae':
        self.env=vae_env()
    if environment=='lstm':
        self.env=lstm_env()
    if environment=='gan':
        self.env=gan_env()
    if environment=='cvae':
        self.env=cvae_env()

  def run(self):
    print("successfully started")
    total_step = 1
    mem = Memory()
          
    while Worker.global_episode < args.max_eps:
      current_state=self.env.reset() #Workers reset each time
      mem.clear()
      ep_reward = 0 #Probably the accumulated reward from episode
      ep_steps = 0
      ep_reward_action=0 #Probably the accumulated reward for each update period
      self.ep_loss = 0

      time_count = 0
      done = False
      while not done: #This is each episode loop, for max_worker_steps iterations
        if ep_steps==max_worker_steps:
           done=True
        
        logits, _ = self.local_model(
            tf.convert_to_tensor(current_state[None, :],
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)[0]
        if ep_steps==0: #First step in episode:
            state_of_action=current_state #First state we are acting from
            action = np.random.choice(self.action_size, p=probs.numpy()) #First action we take
            ep_reward_action=0 #The accumulated reward for update period
        
        if (ep_steps!=0) & (ep_steps % 100==0): #Every 100 steps
            mem.store(state_of_action, action, ep_reward_action)
            print("memory of past 120 steps stored")
            
            with tf.GradientTape() as tape:
              total_loss = self.compute_loss(state_of_action,
                                             mem,
                                             args.gamma,
                                             )

            self.ep_loss += total_loss #Update episode loss with 'period' loss

            # Calculate local gradients
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            # Push local gradients to global model
            self.opt.apply_gradients(zip(grads,
                                         self.global_model.trainable_weights))
            # Update local model with new weights
            self.local_model.set_weights(self.global_model.get_weights())
            mem.clear() #Now we reset the memory
            time_count = 0 #Reset the time count
            #For now, not worrying about saving best model - code is dependent on comparing
            #Period rewards from particular states, and we have too many states
            
            #Now, we need to change the state_of_action
            state_of_action=new_state
            action = np.random.choice(self.action_size, p=probs.numpy())
            print("new action is "+str(action))
            ep_reward_action=0
 
        
            
        #This happens every step
        #rollout_action=np.random.choice(self.action_size, p=probs.numpy())
        new_state= self.env.transition(action) #Transition environment based on most recent action selection
        reward=reward_func(new_state[0])
        if done:
          reward = -1
        
        ep_reward += reward #Each step, update the episode's accumulated reward
        ep_reward_action += reward #Each step, update the 'update period's' accumulated reward
        
        if done:
            print("Saving at EPS",Worker.global_episode)
            Worker.global_moving_average_reward = record(Worker.global_episode, ep_reward, self.worker_idx, Worker.global_moving_average_reward, self.result_queue, self.ep_loss, ep_steps)
            Worker.episode_reward_list.append(ep_reward)
            if ep_reward > Worker.best_score:
                Worker.best_score = ep_reward
                with Worker.save_lock:
                    print("Saving best model to {}, episode score: {}".format(self.save_dir, ep_reward))
                    self.global_model.save_weights(os.path.join(self.save_dir,'model_{}.h5'.format(self.game_name)))
            Worker.global_episode += 1

           
        #This happens every step
        ep_steps += 1
        time_count += 1
        current_state = new_state
        total_step += 1
        
    self.result_queue.put(None) #End the result queue with 'done'

  def compute_loss(self,
                   new_state,
                   memory,
                   gamma=0.99):
    reward_sum = self.local_model(
      tf.convert_to_tensor(new_state[None, :],
                           dtype=tf.float32))[-1].numpy()[0]

    discounted_rewards = []
    for reward in memory.rewards[::-1]:  # reverse buffer r
      reward_sum = reward + gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    logits, values = self.local_model(
        tf.convert_to_tensor(np.vstack(memory.states),
                             dtype=tf.float32))
    #Logits are the predicted action probabilities for past states
    #Values are the predicted rewards
    #Advantage is difference between actual rewards and predicted rewards
    
    # Get our advantages
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                            dtype=tf.float32) - values

    # Value loss
    value_loss = advantage ** 2

    # Calculate our policy loss
    policy = tf.nn.softmax(logits)

    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                 logits=logits)
    
    #Policy loss is difference between actions taken and predicted actions based on states
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss
    

#Train agents
max_worker_steps=500
args.max_eps=2000

Worker.global_episode=0
Worker.global_moving_average_reward = 0
Worker.episode_reward_list=[]
Worker.best_score = 0

VaeMaster2=MasterAgent()
vae_ep_rewards, vae_mov_rewards=VaeMaster2.train(environment='vae')

plt.plot(vae_ep_rewards)
plt.plot(vae_mov_rewards)

VaeMaster2.global_model.save_weights("vae_model_weights_7-1.h5")
np.savetxt("vae_train_ep_rewards_7-1.txt",vae_ep_rewards,delimiter=",")
np.savetxt("vae_train_mov_rewards_7-1.txt",vae_mov_rewards,delimiter=",")

Worker.global_episode=0
Worker.global_moving_average_reward = 0
Worker.episode_reward_list=[]
Worker.best_score = 0

HmmMaster2=MasterAgent()
hmm_ep_rewards, hmm_mov_rewards=HmmMaster2.train(environment='hmm')

plt.plot(hmm_ep_rewards)
plt.plot(hmm_mov_rewards)

HmmMaster2.global_model.save_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_model_weights_1-22_FIXED_second_attempt.h5")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_train_ep_rewards_FIXED_second_attempt.txt",hmm_ep_rewards,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_train_mov_rewards_FIXED_second_attempt.txt",hmm_mov_rewards,delimiter=",")

Worker.global_episode=0
Worker.global_moving_average_reward = 0
Worker.episode_reward_list=[]
Worker.best_score = 0

LstmMaster2=MasterAgent()
lstm_ep_rewards, lstm_mov_rewards=LstmMaster2.train(environment='lstm')
    #mov_avg_rewards is reset every time - save it each time

plt.plot(lstm_ep_rewards)
plt.plot(lstm_mov_rewards)

LstmMaster2.global_model.save_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_model_weights_6-25_2000.h5")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_train_ep_rewards_2000.txt",lstm_ep_rewards,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_train_mov_rewards_2000.txt",lstm_mov_rewards,delimiter=",")

Worker.global_episode=0
Worker.global_moving_average_reward = 0
Worker.episode_reward_list=[]
Worker.best_score = 0

GanMaster2=MasterAgent()
gan_ep_rewards, gan_mov_rewards=GanMaster2.train(environment='gan')

plt.plot(gan_ep_rewards)
plt.plot(gan_mov_rewards)

GanMaster2.global_model.save_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_model_weights_6-25_v2.h5")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_train_ep_rewards.txt",gan_ep_rewards,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_train_mov_rewards.txt",gan_mov_rewards,delimiter=",")

Worker.global_episode=0
Worker.global_moving_average_reward = 0
Worker.episode_reward_list=[]
Worker.best_score = 0

CvaeMaster2=MasterAgent()
cvae_ep_rewards, cvae_mov_rewards=CvaeMaster2.train(environment='cvae')

plt.plot(cvae_ep_rewards)
plt.plot(cvae_mov_rewards)

CvaeMaster2.global_model.save_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_model_weights_6-30_.25.h5")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_train_ep_rewards_.25.txt",cvae_ep_rewards,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_train_mov_rewards_.25.txt",cvae_mov_rewards,delimiter=",")


vae_mov=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_train_mov_rewards.txt",delimiter=",")
hmm_mov=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_train_mov_rewards_FIXED_second_attempt.txt",delimiter=",")
lstm_mov=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_train_mov_rewards_2000.txt",delimiter=",")
gan_mov=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_train_mov_rewards.txt",delimiter=",")
cvae_mov=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_train_mov_rewards_.25.txt",delimiter=",")



#Compare models - make new agents since you re-did 'play' code
vae_player=MasterAgent()
vae_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_model_weights_1-22_v2.h5")

hmm_player=MasterAgent()
hmm_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_model_weights_1-22_FIXED.h5")

lstm_player=MasterAgent()
lstm_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_model_weights_6-25_2000.h5")

gan_player=MasterAgent()
gan_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_model_weights_6-25_v2.h5")

cvae_player=MasterAgent()
cvae_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_model_weights_6-30_.25.h5")

vae_player.play(steps=24*7,trials=1000,stochastic=True,environment='vae') 
vae_on_vae=vae_player.simulated_discounted_rewards

vae_player.play(steps=24*7,trials=1000,stochastic=True,environment='hmm') 
vae_on_hmm=vae_player.simulated_discounted_rewards

vae_player.play(steps=24*7,trials=1000,stochastic=True,environment='lstm') 
vae_on_lstm=vae_player.simulated_discounted_rewards

vae_player.play(steps=24*7,trials=1000,stochastic=True,environment='gan') 
vae_on_gan=vae_player.simulated_discounted_rewards

vae_player.play(steps=24*7,trials=1000,stochastic=True,environment='cvae') 
vae_on_cvae=vae_player.simulated_discounted_rewards

hmm_player.play(steps=24*7,trials=1000,stochastic=True,environment='vae') 
hmm_on_vae=hmm_player.simulated_discounted_rewards

hmm_player.play(steps=24*7,trials=1000,stochastic=True,environment='hmm') 
hmm_on_hmm=hmm_player.simulated_discounted_rewards
    
hmm_player.play(steps=24*7,trials=1000,stochastic=True,environment='lstm') 
hmm_on_lstm=hmm_player.simulated_discounted_rewards

hmm_player.play(steps=24*7,trials=1000,stochastic=True,environment='gan') 
hmm_on_gan=hmm_player.simulated_discounted_rewards

hmm_player.play(steps=24*7,trials=1000,stochastic=True,environment='cvae') 
hmm_on_cvae=hmm_player.simulated_discounted_rewards

lstm_player.play(steps=24*7,trials=1000,stochastic=True,environment='vae') #DONE
lstm_on_vae=lstm_player.simulated_discounted_rewards

lstm_player.play(steps=24*7,trials=1000,stochastic=True,environment='hmm') 
lstm_on_hmm=lstm_player.simulated_discounted_rewards

lstm_player.play(steps=24*7,trials=1000,stochastic=True,environment='lstm') #Done
lstm_on_lstm=lstm_player.simulated_discounted_rewards

lstm_player.play(steps=24*7,trials=1000,stochastic=True,environment='gan') #DONE
lstm_on_gan=lstm_player.simulated_discounted_rewards

lstm_player.play(steps=24*7,trials=1000,stochastic=True,environment='cvae') 
lstm_on_cvae=lstm_player.simulated_discounted_rewards

gan_player.play(steps=24*7,trials=1000,stochastic=True,environment='vae') #Done
gan_on_vae=gan_player.simulated_discounted_rewards

gan_player.play(steps=24*7,trials=1000,stochastic=True,environment='hmm') 
gan_on_hmm=gan_player.simulated_discounted_rewards

gan_player.play(steps=24*7,trials=1000,stochastic=True,environment='lstm') #Done
gan_on_lstm=gan_player.simulated_discounted_rewards

gan_player.play(steps=24*7,trials=1000,stochastic=True,environment='gan') #Done
gan_on_gan=gan_player.simulated_discounted_rewards

gan_player.play(steps=24*7,trials=1000,stochastic=True,environment='cvae') 
gan_on_cvae=gan_player.simulated_discounted_rewards

cvae_player.play(steps=24*7,trials=1000,stochastic=True,environment='vae') 
cvae_on_vae=cvae_player.simulated_discounted_rewards

cvae_player.play(steps=24*7,trials=1000,stochastic=True,environment='hmm') 
cvae_on_hmm=cvae_player.simulated_discounted_rewards

cvae_player.play(steps=24*7,trials=1000,stochastic=True,environment='lstm') 
cvae_on_lstm=cvae_player.simulated_discounted_rewards

cvae_player.play(steps=24*7,trials=1000,stochastic=True,environment='gan') 
cvae_on_gan=cvae_player.simulated_discounted_rewards

cvae_player.play(steps=24*7,trials=1000,stochastic=True,environment='cvae') 
cvae_on_cvae=cvae_player.simulated_discounted_rewards

#Should be ~29
np.mean(vae_player.simulated_discounted_rewards)

    #CREATE FLOOR
player=MasterAgent() #VAE env floor is 15.5538
player.play_untreated(steps=24*7,trials=1000,environment='vae')
np.mean(player.simulated_discounted_rewards)

player=MasterAgent() #HMM floor is -8.0187
player.play_untreated(steps=24*7,trials=1000,environment='hmm')
np.mean(player.simulated_discounted_rewards)

player=MasterAgent() #LSTM floor is -35.01
player.play_untreated(steps=24*7,trials=10,environment='lstm') #Remember LSTM does batches of 100
np.mean(player.simulated_discounted_rewards)

player=MasterAgent() #GAN floor is -53.35
player.play_untreated(steps=24*7,trials=1000,environment='gan')
np.mean(player.simulated_discounted_rewards)

player=MasterAgent() #CVAE floor is -8.614
player.play_untreated(steps=24*7,trials=1000,environment='cvae')
np.mean(player.simulated_discounted_rewards)



#SAVE RESULTS
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_vae_rewards_KL1.txt",vae_on_vae,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_hmm_rewards_KL1.txt",vae_on_hmm,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_lstm_rewards_KL1.txt",vae_on_lstm,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_gan_rewards_KL1.txt",vae_on_gan,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_cvae_rewards_KL1.txt",vae_on_cvae,delimiter=",")

np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_vae_rewards_1-22.txt",hmm_on_vae,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_hmm_rewards_1-22.txt",hmm_on_hmm,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_lstm_rewards_6-26.txt",hmm_on_lstm,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_gan_rewards_6-26.txt",hmm_on_gan,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_cvae_rewards_6-26.txt",hmm_on_cvae,delimiter=",")

#np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_vae_rewards_6-26.txt",lstm_on_vae,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_hmm_rewards_6-26.txt",lstm_on_hmm,delimiter=",")
#np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_lstm_rewards_6-26.txt",lstm_on_lstm,delimiter=",")
#np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_gan_rewards_6-26.txt",lstm_on_gan,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_cvae_rewards_6-26.txt",lstm_on_cvae,delimiter=",")

#np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_vae_rewards_6-26.txt",gan_on_vae,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_hmm_rewards_6-26.txt",gan_on_hmm,delimiter=",")
#np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_lstm_rewards_6-26.txt",gan_on_lstm,delimiter=",")
#np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_gan_rewards_6-26.txt",gan_on_gan,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_cvae_rewards_6-26.txt",gan_on_cvae,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_vae_rewards_6-26.txt",cvae_on_vae,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_hmm_rewards_6-26.txt",cvae_on_hmm,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_lstm_rewards_6-26.txt",cvae_on_lstm,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_gan_rewards_6-26.txt",cvae_on_gan,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_cvae_rewards_6-26.txt",cvae_on_cvae,delimiter=",")



#OFFICIAL RL ANALYSIS
vae_on_vae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_vae_rewards_1-22.txt",delimiter=",")
vae_on_hmm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_hmm_rewards_1-22.txt",delimiter=",")
vae_on_lstm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_lstm_rewards_6-26.txt",delimiter=",")
vae_on_gan=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_gan_rewards_6-26.txt",delimiter=",")
vae_on_cvae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_on_cvae_rewards_6-26.txt",delimiter=",")

hmm_on_vae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_vae_rewards_1-22.txt",delimiter=",")
hmm_on_hmm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_hmm_rewards_1-22.txt",delimiter=",")
hmm_on_lstm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_lstm_rewards_6-26.txt",delimiter=",")
hmm_on_gan=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_gan_rewards_6-26.txt",delimiter=",")
hmm_on_cvae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_on_cvae_rewards_6-26.txt",delimiter=",")

lstm_on_vae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_vae_rewards_6-26.txt",delimiter=",")
lstm_on_hmm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_hmm_rewards_6-26.txt",delimiter=",")
lstm_on_lstm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_lstm_rewards_6-26.txt",delimiter=",")
lstm_on_gan=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_gan_rewards_6-26.txt",delimiter=",")
lstm_on_cvae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_on_cvae_rewards_6-26.txt",delimiter=",")

gan_on_vae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_vae_rewards_6-26.txt",delimiter=",")
gan_on_hmm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_hmm_rewards_6-26.txt",delimiter=",")
gan_on_lstm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_lstm_rewards_6-26.txt",delimiter=",")
gan_on_gan=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_gan_rewards_6-26.txt",delimiter=",")
gan_on_cvae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_on_cvae_rewards_6-26.txt",delimiter=",")

cvae_on_vae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_vae_rewards_6-26.txt",delimiter=",")
cvae_on_hmm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_hmm_rewards_6-26.txt",delimiter=",")
cvae_on_lstm=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_lstm_rewards_6-26.txt",delimiter=",")
cvae_on_gan=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_gan_rewards_6-26.txt",delimiter=",")
cvae_on_cvae=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_on_cvae_rewards_6-26.txt",delimiter=",")

    #LIMIT ALL TO 1000 EPISODES
vae_on_vae=vae_on_vae[0:1000];vae_on_hmm=vae_on_hmm[0:1000];hmm_on_vae=hmm_on_vae[0:1000];hmm_on_hmm=hmm_on_hmm[0:1000]

floors=[15.5538,-8.0187,-35.01,-53.35,-8.614]

vae_all=np.concatenate([vae_on_vae-floors[0],vae_on_hmm-floors[1],vae_on_lstm-floors[2],vae_on_gan-floors[3],vae_on_cvae-floors[4]])
hmm_all=np.concatenate([hmm_on_vae-floors[0],hmm_on_hmm-floors[1],hmm_on_lstm-floors[2],hmm_on_gan-floors[3],hmm_on_cvae-floors[4]])
lstm_all=np.concatenate([lstm_on_vae-floors[0],lstm_on_hmm-floors[1],lstm_on_lstm-floors[2],lstm_on_gan-floors[3],lstm_on_cvae-floors[4]])
gan_all=np.concatenate([gan_on_vae-floors[0],gan_on_hmm-floors[1],gan_on_lstm-floors[2],gan_on_gan-floors[3],gan_on_cvae-floors[4]])
cvae_all=np.concatenate([cvae_on_vae-floors[0],cvae_on_hmm-floors[1],cvae_on_lstm-floors[2],cvae_on_gan-floors[3],cvae_on_cvae-floors[4]])

np.mean(vae_all)
np.mean(hmm_all)
np.mean(lstm_all)
np.mean(gan_all)
np.mean(cvae_all)

plt.hist(vae_all)
plt.hist(hmm_all)
plt.hist(lstm_all)
plt.hist(gan_all)
plt.hist(cvae_all)

vae_se=1.96*np.std(vae_all)/np.sqrt(len(vae_all))
hmm_se=1.96*np.std(hmm_all)/np.sqrt(len(hmm_all))
lstm_se=1.96*np.std(lstm_all)/np.sqrt(len(lstm_all))
gan_se=1.96*np.std(gan_all)/np.sqrt(len(gan_all))
cvae_se=1.96*np.std(cvae_all)/np.sqrt(len(cvae_all))

plt.figure(figsize=(6,3))
#plt.plot([np.mean(cvae_all),np.mean(vae_all),np.mean(hmm_all),np.mean(lstm_all),np.mean(gan_all)],'o')
plt.bar(x=['CVAE','tVAE','POMDP','LSTM','CGAN'],height=[np.mean(cvae_all),np.mean(vae_all),np.mean(hmm_all),np.mean(lstm_all),np.mean(gan_all)],color=".5",width=.5)
plt.errorbar(x=['CVAE','tVAE','POMDP','LSTM','CGAN'],y=[np.mean(cvae_all),np.mean(vae_all),np.mean(hmm_all),np.mean(lstm_all),np.mean(gan_all)],
                yerr=[vae_se,hmm_se,lstm_se,gan_se,cvae_se],fmt='o',color="0",capsize=12)
plt.yticks([0,2,4,6,8,10,12])


np.mean(np.concatenate([vae_on_lstm,vae_on_gan,vae_on_cvae]))
np.mean(np.concatenate([hmm_on_lstm,hmm_on_gan,hmm_on_cvae]))

np.mean(np.concatenate([vae_on_hmm,vae_on_gan,vae_on_cvae]))
np.mean(np.concatenate([lstm_on_hmm,lstm_on_gan,lstm_on_cvae]))

np.mean(np.concatenate([vae_on_hmm,vae_on_lstm,vae_on_cvae]))
np.mean(np.concatenate([gan_on_hmm,gan_on_lstm,gan_on_cvae]))

np.mean(np.concatenate([vae_on_hmm,vae_on_lstm,vae_on_gan]))
np.mean(np.concatenate([cvae_on_hmm,cvae_on_lstm,cvae_on_gan]))

np.mean(np.concatenate(vae_on_vae,vae_on_hmm,vae_on_lstm,vae_on_gan,vae_on_cvae))



np.mean(vae_on_vae)
np.mean(hmm_on_vae)
np.mean(lstm_on_vae)
np.mean(gan_on_vae)
np.mean(cvae_on_vae)

plt.hist(vae_on_hmm,fill=False,histtype="step",stacked=True,color="black")
plt.hist(hmm_on_hmm,fill=False,histtype="step",stacked=True,color="red")
plt.hist(lstm_on_hmm,fill=False,histtype="step",stacked=True,color="blue")
plt.hist(gan_on_hmm,fill=False,histtype="step",stacked=True,color="orange")
plt.hist(cvae_on_hmm,fill=False,histtype="step",stacked=True,color="green")

np.mean(vae_on_hmm)
np.mean(hmm_on_hmm)
np.mean(lstm_on_hmm)
np.mean(gan_on_hmm)
np.mean(cvae_on_hmm)

plt.hist(vae_on_lstm,fill=False,histtype="step",stacked=True,color="black")
plt.hist(hmm_on_lstm,fill=False,histtype="step",stacked=True,color="red")
plt.hist(lstm_on_lstm,fill=False,histtype="step",stacked=True,color="blue")
plt.hist(gan_on_lstm,fill=False,histtype="step",stacked=True,color="orange")
plt.hist(cvae_on_lstm,fill=False,histtype="step",stacked=True,color="green")

np.mean(vae_on_lstm)
np.mean(hmm_on_lstm)
np.mean(lstm_on_lstm)
np.mean(gan_on_lstm)
np.mean(cvae_on_lstm)

plt.hist(vae_on_gan,fill=False,histtype="step",stacked=True,color="black")
plt.hist(hmm_on_gan,fill=False,histtype="step",stacked=True,color="red")
plt.hist(lstm_on_gan,fill=False,histtype="step",stacked=True,color="blue")
plt.hist(gan_on_gan,fill=False,histtype="step",stacked=True,color="orange")
plt.hist(cvae_on_gan,fill=False,histtype="step",stacked=True,color="green")

np.mean(vae_on_gan)
np.mean(hmm_on_gan)
np.mean(lstm_on_gan)
np.mean(gan_on_gan)
np.mean(cvae_on_gan)

plt.hist(vae_on_cvae,fill=False,histtype="step",stacked=True,color="black")
plt.hist(hmm_on_cvae,fill=False,histtype="step",stacked=True,color="red")
plt.hist(lstm_on_cvae,fill=False,histtype="step",stacked=True,color="blue")
plt.hist(gan_on_cvae,fill=False,histtype="step",stacked=True,color="orange")
plt.hist(cvae_on_cvae,fill=False,histtype="step",stacked=True,color="green")

np.mean(vae_on_cvae)
np.mean(hmm_on_cvae)
np.mean(lstm_on_cvae)
np.mean(gan_on_cvae)
np.mean(cvae_on_cvae)


np.mean(np.concatenate([vae_on_lstm,vae_on_gan,vae_on_cvae]))
np.mean(np.concatenate([hmm_on_lstm,hmm_on_gan,hmm_on_cvae]))

np.mean(np.concatenate([vae_on_hmm,vae_on_gan,vae_on_cvae]))
np.mean(np.concatenate([lstm_on_hmm,lstm_on_gan,lstm_on_cvae]))

np.mean(np.concatenate([vae_on_hmm,vae_on_lstm,vae_on_cvae]))
np.mean(np.concatenate([gan_on_hmm,gan_on_lstm,gan_on_cvae]))

np.mean(np.concatenate([vae_on_hmm,vae_on_lstm,vae_on_gan]))
np.mean(np.concatenate([cvae_on_hmm,cvae_on_lstm,cvae_on_gan]))

#OFFICIAL POLICY ANALYSIS
#Read in fulldata
fulldata=pd.read_csv("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Nemani Patient Data Cleaned Imputed.csv")
ind=np.unique(fulldata.INDEX)
#fulldata.aPTT=(fulldata.aPTT-z_means[0])/z_std[0]

     #Load policy weights
vae_player=MasterAgent()
vae_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/vae_model_weights_1-22_v2.h5")

hmm_player=MasterAgent()
hmm_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/hmm_model_weights_1-22_FIXED.h5")

lstm_player=MasterAgent()
lstm_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/lstm_model_weights_6-25_2000.h5")

gan_player=MasterAgent()
gan_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/gan_model_weights_6-25_v2.h5")

cvae_player=MasterAgent()
cvae_player.global_model.load_weights("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/RL Model Weights/cvae_model_weights_6-30_.25.h5")

vae_w=vae_player.global_model.get_weights()
hmm_w=hmm_player.global_model.get_weights()
lstm_w=lstm_player.global_model.get_weights()
gan_w=gan_player.global_model.get_weights()
cvae_w=cvae_player.global_model.get_weights()

state_data=fulldata[["aPTT","CO2_sig","HR_sig","creat_sig","gcs_sig","hematocrit_sig","hemoglobin_sig","inr_sig","plat_sig","pt_sig","spo2_sig","temp_sig","urea_sig","wbc_sig"]]
state_data=(state_data-z_means)/z_std

vae_policy=vae_player.global_model(tf.convert_to_tensor(np.array(state_data), dtype=tf.float32))[0]
hmm_policy=hmm_player.global_model(tf.convert_to_tensor(np.array(state_data), dtype=tf.float32))[0]
lstm_policy=lstm_player.global_model(tf.convert_to_tensor(np.array(state_data), dtype=tf.float32))[0]
gan_policy=gan_player.global_model(tf.convert_to_tensor(np.array(state_data), dtype=tf.float32))[0]
cvae_policy=cvae_player.global_model(tf.convert_to_tensor(np.array(state_data), dtype=tf.float32))[0]

vae_policy=tf.nn.softmax(vae_policy,axis=1).numpy()
hmm_policy=tf.nn.softmax(hmm_policy,axis=1).numpy()
lstm_policy=tf.nn.softmax(lstm_policy,axis=1).numpy()
gan_policy=tf.nn.softmax(gan_policy,axis=1).numpy()
cvae_policy=tf.nn.softmax(cvae_policy,axis=1).numpy()

np.mean(vae_policy,axis=0)
np.mean(hmm_policy,axis=0)
np.mean(lstm_policy,axis=0)
np.mean(gan_policy,axis=0)
np.mean(cvae_policy,axis=0)


vae_high=np.sum(vae_policy[:,3:],axis=1)
vae_low=np.sum(vae_policy[:,0:2],axis=1)
hmm_high=np.sum(hmm_policy[:,3:],axis=1)
hmm_low=np.sum(hmm_policy[:,0:2],axis=1)
lstm_high=np.sum(lstm_policy[:,5:],axis=1)
lstm_low=np.sum(lstm_policy[:,0:4],axis=1)
cvae_high=np.sum(cvae_policy[:,3:],axis=1)
cvae_low=np.sum(cvae_policy[:,0:2],axis=1)

np.mean(fulldata.aPTT[vae_high>=np.quantile(vae_high,.9)])
np.mean(fulldata.aPTT[vae_low>=np.quantile(vae_low,.9)])

np.mean(fulldata.aPTT[hmm_high>=np.quantile(hmm_high,.8)])
np.mean(fulldata.aPTT[hmm_low>=np.quantile(hmm_low,.8)])

np.mean(fulldata.aPTT[lstm_high>=np.quantile(lstm_high,.8)])
np.mean(fulldata.aPTT[lstm_low>=np.quantile(lstm_low,.8)])

np.mean(fulldata.aPTT[gan_high>=np.quantile(gan_high,.8)])
np.mean(fulldata.aPTT[gan_low>=np.quantile(gan_low,.8)])

np.mean(fulldata.aPTT[cvae_high>=np.quantile(cvae_high,.8)])
np.mean(fulldata.aPTT[cvae_low>=np.quantile(cvae_low,.8)])


vae_avg=np.sum(vae_policy*np.array([0,1,2,3,4,5]),axis=1)
hmm_avg=np.sum(hmm_policy*np.array([0,1,2,3,4,5]),axis=1)
lstm_avg=np.sum(lstm_policy*np.array([0,1,2,3,4,5]),axis=1)
gan_avg=np.sum(gan_policy*np.array([0,1,2,3,4,5]),axis=1)
cvae_avg=np.sum(cvae_policy*np.array([0,1,2,3,4,5]),axis=1)

aptt_dist=(fulldata.aPTT-60)*(fulldata.aPTT<60)+(fulldata.aPTT-100)*(fulldata.aPTT>100)

np.corrcoef(aptt_dist,vae_avg)
np.corrcoef(aptt_dist,hmm_avg)
np.corrcoef(aptt_dist,lstm_avg)
np.corrcoef(aptt_dist,gan_avg)
np.corrcoef(aptt_dist,cvae_avg)


np.corrcoef(aptt_dist,vae_policy[:,5])

#Read in fulldata
fulldata=pd.read_csv("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Nemani Patient Data Cleaned Imputed.csv")
ind=np.unique(fulldata.INDEX)
fulldata.aPTT=(fulldata.aPTT-z_means[0])/z_std[0]

#Evaluate policy return for difference dosages
mean_hep=[]

counter=0
for i in ind:
    counter=counter+1
    if counter%100==0:
        print(counter)
    seq=fulldata[fulldata.INDEX==i][["Heparin_1","Heparin_2","Heparin_3","Heparin_4","Heparin_5","Heparin_6"]]
    hepseq=seq.Heparin_1*0+seq.Heparin_2*1+seq.Heparin_3*2+seq.Heparin_4*3+seq.Heparin_5*4+seq.Heparin_6*5
    mean_hep.append(np.mean(hepseq))

plt.hist(mean_hep)

plt.hist(np.around(mean_hep))
    #OFFICIAL RESULT: MORE PATIENTS RECEIVE AVERAGE DOSES OF 2 THAN 4. 2 BETTER MATCHES ACTUAL PATIENT TRAJECTORIES
    
     #Policy examples for individual patients
vae_rec=[]
hmm_rec=[]
lstm_rec=[]
gan_rec=[]
cvae_rec=[]

states=[]

for i in range(2067):
    print(i)
    patseq=fulldata[fulldata.INDEX==ind[i]][["aPTT","CO2_sig","HR_sig","creat_sig","gcs_sig","hematocrit_sig","hemoglobin_sig","inr_sig","plat_sig","pt_sig","spo2_sig","temp_sig","urea_sig","wbc_sig"]]
    hepseq=fulldata[fulldata.INDEX==ind[i]][["Heparin_1","Heparin_2","Heparin_3","Heparin_4","Heparin_5","Heparin_6"]]
    real_hep=1*hepseq.Heparin_2+2*hepseq.Heparin_3+3*hepseq.Heparin_4+4*hepseq.Heparin_5+5*hepseq.Heparin_6
    
    for it in range(50):
        states.append(np.array(patseq))
            
states=np.concatenate(states,axis=0)

for i in range(2067):
    print(i)
    patseq=fulldata[fulldata.INDEX==ind[i]][["aPTT","CO2_sig","HR_sig","creat_sig","gcs_sig","hematocrit_sig","hemoglobin_sig","inr_sig","plat_sig","pt_sig","spo2_sig","temp_sig","urea_sig","wbc_sig"]]
    hepseq=fulldata[fulldata.INDEX==ind[i]][["Heparin_1","Heparin_2","Heparin_3","Heparin_4","Heparin_5","Heparin_6"]]
    real_hep=1*hepseq.Heparin_2+2*hepseq.Heparin_3+3*hepseq.Heparin_4+4*hepseq.Heparin_5+5*hepseq.Heparin_6
    
    vae_hep=[]
    hmm_hep=[]
    lstm_hep=[]
    gan_hep=[]
    cvae_hep=[]
    
    for it in range(50):
        vae_seq=[]
        hmm_seq=[]
        lstm_seq=[]
        gan_seq=[]
        cvae_seq=[]
        for step in range(patseq.shape[0]):
            state=np.array(patseq.iloc[step,:])
            vae_policy, value = vae_player.global_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            vae_policy = tf.nn.softmax(vae_policy)
            vae_action=np.random.choice([0,1,2,3,4,5],p=np.array(vae_policy[0]))
          
            hmm_policy, value = hmm_player.global_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            hmm_policy = tf.nn.softmax(hmm_policy)
            hmm_action=np.random.choice([0,1,2,3,4,5],p=np.array(hmm_policy[0]))                
    
            lstm_policy, value = lstm_player.global_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            lstm_policy = tf.nn.softmax(lstm_policy)
            lstm_action=np.random.choice([0,1,2,3,4,5],p=np.array(lstm_policy[0]))
    
            gan_policy, value = gan_player.global_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            gan_policy = tf.nn.softmax(gan_policy)
            gan_action=np.random.choice([0,1,2,3,4,5],p=np.array(gan_policy[0]))
            
            cvae_policy, value = cvae_player.global_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            cvae_policy = tf.nn.softmax(cvae_policy)
            cvae_action=np.random.choice([0,1,2,3,4,5],p=np.array(cvae_policy[0]))
                        
            vae_seq.append(vae_action)#+np.random.normal(scale=.1))
            hmm_seq.append(hmm_action)#+np.random.normal(scale=.1))
            lstm_seq.append(lstm_action)#+np.random.normal(scale=.1))
            gan_seq.append(gan_action)
            cvae_seq.append(cvae_action)
            
        vae_hep.append(vae_seq)
        hmm_hep.append(hmm_seq)
        lstm_hep.append(lstm_seq)
        gan_hep.append(gan_seq)
        cvae_hep.append(cvae_seq)
    
    vae_hep=np.array(vae_hep).reshape(-1)
    hmm_hep=np.array(hmm_hep).reshape(-1)
    lstm_hep=np.array(lstm_hep).reshape(-1)
    gan_hep=np.array(gan_hep).reshape(-1)
    cvae_hep=np.array(cvae_hep).reshape(-1)
    
    vae_rec.append(vae_hep)
    hmm_rec.append(hmm_hep)
    lstm_rec.append(lstm_hep)
    gan_rec.append(gan_hep)
    cvae_rec.append(cvae_hep)
    
vae_rec_final=np.concatenate(vae_rec)
hmm_rec_final=np.concatenate(hmm_rec)
lstm_rec_final=np.concatenate(lstm_rec)
gan_rec_final=np.concatenate(gan_rec)
cvae_rec_final=np.concatenate(cvae_rec)

np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/vae_actions.txt",vae_rec_final,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/hmm_actions.txt",hmm_rec_final,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/lstm_actions.txt",lstm_rec_final,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/gan_actions.txt",gan_rec_final,delimiter=",")
np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/cvae_actions.txt",cvae_rec_final,delimiter=",")

#Analyze recommendations
vae_rec_final=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/vae_actions.txt",delimiter=",")
hmm_rec_final=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/hmm_actions.txt",delimiter=",")
lstm_rec_final=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/lstm_actions.txt",delimiter=",")
gan_rec_final=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/gan_actions.txt",delimiter=",")
cvae_rec_final=np.loadtxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Policy Analysis/cvae_actions.txt",delimiter=",")

np.mean(vae_rec_final==2)
np.mean(hmm_rec_final==2)
np.mean(lstm_rec_final==4)
np.mean(gan_rec_final==0)
np.mean(cvae_rec_final==2)


plt.hist(vae_rec_final)
plt.hist(hmm_rec_final)
plt.hist(lstm_rec_final)
plt.hist(gan_rec_final)
plt.hist(cvae_rec_final)






#OFFICIAL TRAJECTORY HISTOGRAMS
vae_ex=np.array(mse_eval(ind[30],environment="vae",ntrials=50,ret="trajectory"))
hmm_ex=np.array(mse_eval(ind[30],environment="hmm",ntrials=50,ret="trajectory"))

plt.plot(np.transpose(vae_ex),color='0',linestyle='dashed')
plt.ylim([20,150])
plt.show()

plt.plot(np.transpose(hmm_ex),color='0',linestyle='dashed')
plt.ylim([20,150])
plt.show()

plt.plot(fulldata[fulldata.INDEX==ind[30]]["aPTT"])
plt.ylim([20,150])
plt.show()


for i in range(len(ind)):
    print(i/len(ind))
    vae_seq=np.array(mse_eval(ind[i],environment="vae",ntrials=100,ret="trajectory"))
    hmm_seq=np.array(mse_eval(ind[i],environment="hmm",ntrials=100,ret="trajectory"))
    np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Alternate Trajectories/VAE/patient"+str(i)+".txt",vae_seq,delimiter=",")
    np.savetxt("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Alternate Trajectories/HMM/patient"+str(i)+".txt",hmm_seq,delimiter=",")


#Need this for results to work
fulldata=pd.read_csv("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Nemani Patient Data Cleaned Imputed.csv")

#Trajectory generation function
import scipy.stats
def mse_eval(index,environment,ntrials,ret="mse"):
    subset=fulldata[fulldata.INDEX==index]
    real_aptt=list(subset.aPTT)[-1]
    
    action_list=list(subset.Heparin_1*0+subset.Heparin_2*1+subset.Heparin_3*2+subset.Heparin_4*3+subset.Heparin_5*4+subset.Heparin_6*5)
    
    subset=np.array(subset[["aPTT","CO2_sig","HR_sig","creat_sig","gcs_sig","hematocrit_sig","hemoglobin_sig","inr_sig","plat_sig","pt_sig","spo2_sig","temp_sig","urea_sig","wbc_sig"]])
    subset=(subset-z_means)/z_std

    first_state=subset[0,:]

    if environment=="hmm":
        env=hmm_env()
        mvars=[scipy.stats.multivariate_normal(mean=hmm_means[i,:],cov=np.diag(hmm_vars[i,:])) for i in range(9)]
        initprobs=[var.pdf(first_state) for var in mvars]
        initprobs=initprobs*env.startprob
        initprobs=initprobs/np.sum(initprobs)
        
        final_aptt=[]
        trajectories=[]
        for i in range(ntrials):
            env.state=np.random.choice(range(9),p=initprobs)
            aptt_meas=[first_state[0]]
            for step in range(len(action_list)-1):
                aptt_meas.append(env.transition(action=action_list[step])[0])
            final_aptt.append((aptt_meas[-1]*z_std[0])+z_means[0])
            aptt_meas=[aptt*z_std[0]+z_means[0] for aptt in aptt_meas]
            trajectories.append(aptt_meas)
            
    if environment=="vae":
        final_aptt=[]
        trajectories=[]
        for i in range(ntrials):
            env=vae_env()
            env.state=first_state.reshape(1,14)
            aptt_meas=[first_state[0]]
            for step in range(len(action_list)-1):
                aptt_meas.append(env.transition(action=action_list[step])[0])
            final_aptt.append((aptt_meas[-1]*z_std[0])+z_means[0])
            aptt_meas=[aptt*z_std[0]+z_means[0] for aptt in aptt_meas]
            trajectories.append(aptt_meas)

    if environment=="lstm":
        trajectories=[]
        first_state=np.array([first_state]*ntrials)
        
        env=lstm_env()
        env.state=first_state;env.history=np.array([]).reshape([ntrials,0,20])
        aptt_meas=env.state[:,0].reshape((ntrials,1))
        
        for step in range(len(action_list)-1):
            env.multi_transition(action=action_list[step])
            aptt_meas=np.concatenate([aptt_meas,env.state[:,0].reshape((ntrials,1))],axis=1)
        
        final_aptt=[aptt_meas[i,-1]*z_std[0]+z_means[0] for i in range(ntrials)]
        aptt_meas=aptt_meas*z_std[0]+z_means[0]
        trajectories=[aptt_meas[i,:] for i in range(ntrials)]
        
    if environment=="gan":
        final_aptt=[]
        trajectories=[]
        for i in range(ntrials):
            env=gan_env()
            env.state=first_state.reshape(1,14)
            aptt_meas=[first_state[0]]
            for step in range(len(action_list)-1):
                aptt_meas.append(env.transition(action=action_list[step])[0])
            final_aptt.append((aptt_meas[-1]*z_std[0])+z_means[0])
            aptt_meas=[aptt*z_std[0]+z_means[0] for aptt in aptt_meas]
            trajectories.append(aptt_meas)
    
    mse=np.mean((np.array(final_aptt)-real_aptt)**2)
    if ret=="mse":
        return mse
    if ret=="aptt":
        return final_aptt
    if ret=='trajectory':
        return trajectories    



#OFFICIAL TRAJECTORY PLOTS
ind=np.unique(fulldata.INDEX)

def trajectory(index,environment):
    subset=fulldata[fulldata.INDEX==index]
    action_list=list(subset.Heparin_1*0+subset.Heparin_2*1+subset.Heparin_3*2+subset.Heparin_4*3+subset.Heparin_5*4+subset.Heparin_6*5)
    
    subset=np.array(subset[["aPTT","CO2_sig","HR_sig","creat_sig","gcs_sig","hematocrit_sig","hemoglobin_sig","inr_sig","plat_sig","pt_sig","spo2_sig","temp_sig","urea_sig","wbc_sig"]])
    subset=(subset-z_means)/z_std

    first_state=subset[0,:]

    if environment=="hmm":
        env=hmm_env()
        mvars=[scipy.stats.multivariate_normal(mean=hmm_means[i,:],cov=np.diag(hmm_vars[i,:])) for i in range(9)]
        initprobs=[var.pdf(first_state) for var in mvars]
        initprobs=initprobs*env.startprob
        initprobs=initprobs/np.sum(initprobs)
        
        aptt_val=[]
        env.state=np.random.choice(range(9),p=initprobs)
        aptt_val.append(first_state[0])
        for step in range(len(action_list)-1):
            aptt_val.append(env.transition(action=action_list[step])[0])
    
    if environment=="vae":
        env=vae_env()
        aptt_val=[]
        env.state=first_state.reshape(1,14)
        aptt_val.append(first_state[0])
        for step in range(len(action_list)-1):
            aptt_val.append(env.transition(action=action_list[step])[0])
    return aptt_val


#Not sure if using this code


    #Plot consecutive entries
vae_first=np.concatenate([seqlist_vae[i][:,0:-1].reshape(-1) for i in range(2067)])
vae_next=np.concatenate([seqlist_vae[i][:,1:].reshape(-1) for i in range(2067)])
mv, bv=np.polyfit(vae_first,vae_next,1)

hmm_first=np.concatenate([seqlist_hmm[i][:,0:-1].reshape(-1) for i in range(2067)])
hmm_next=np.concatenate([seqlist_hmm[i][:,1:].reshape(-1) for i in range(2067)])
mh, bh=np.polyfit(hmm_first,hmm_next,1)

select_idx=np.random.choice(np.array(range(len(vae_first))),10000,replace=False)

plt.plot(vae_first[select_idx],vae_next[select_idx],'o')
plt.plot(vae_first[select_idx],vae_first[select_idx]*mv+bv)
plt.xlim([20,150])
plt.ylim([20,150])
plt.show()

plt.plot(hmm_first[select_idx],hmm_next[select_idx],'o')
plt.plot(hmm_first[select_idx],hmm_first[select_idx]*mh+bh)
plt.xlim([20,150])
plt.ylim([20,150])
plt.show()

np.corrcoef(vae_first,vae_next)
np.corrcoef(hmm_first,hmm_next)

    

    #Final aPTT mae
final_real=[list(fulldata[fulldata.INDEX==i]["aPTT"])[-1] for i in ind]

mae_vae=[np.mean(np.absolute(seqlist_vae[i][:,-1]-final_real[i])) for i in range(len(ind))]
hmm_vae=[np.mean(np.absolute(seqlist_hmm[i][:,-1]-final_real[i])) for i in range(len(ind))]




