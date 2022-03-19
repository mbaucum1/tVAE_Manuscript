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

dataX_trim=pd.read_csv("dataX.csv")
dataY_trim=pd.read_csv("dataY.csv")

dataX_trim=np.array(dataX_trim.drop("Unnamed: 0",axis=1))
dataY_trim=np.array(dataY_trim.drop("Unnamed: 0",axis=1))

#Read in means/std (these haven't changed, despite change to HMM method)
z_means=np.loadtxt("z_means.txt",delimiter=",")
z_std=np.loadtxt("z_std.txt",delimiter=",")

fulldata=pd.read_csv("C:/Users/bauc9/OneDrive - University of Tennessee/Documents/Projects/VAE for RL/Nemani Patient Data Cleaned Imputed.csv")
var_min=np.min(dataY_trim,axis=0)
var_max=np.max(dataY_trim,axis=0)

def reward_func(aPTT):
    new_aptt=aPTT*z_std[0]+z_means[0]
    reward_val=2/(1+np.exp(-(new_aptt-60)))-2/(1+np.exp(-(new_aptt-100)))-1
    return reward_val

#Create HMM environment
hep1_trans=np.loadtxt("hep1_transmat_TRAIN.txt",delimiter=",")
hep2_trans=np.loadtxt("hep2_transmat_TRAIN.txt",delimiter=",")
hep3_trans=np.loadtxt("hep3_transmat_TRAIN.txt",delimiter=",")
hep4_trans=np.loadtxt("hep4_transmat_TRAIN.txt",delimiter=",")
hep5_trans=np.loadtxt("hep5_transmat_TRAIN.txt",delimiter=",")
hep6_trans=np.loadtxt("hep6_transmat_TRAIN.txt",delimiter=",")

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
encoder1_weights=np.loadtxt('encoder1_weights_TRAIN_kl5.txt',delimiter=',')
encoder1_bias=np.loadtxt('encoder1_bias_TRAIN_kl5.txt',delimiter=',')
z_mean_weights=np.loadtxt('z_mean_weights_TRAIN_kl5.txt',delimiter=',')
z_mean_bias=np.loadtxt('z_mean_bias_TRAIN_kl5.txt',delimiter=',')
z_logvar_weights=np.loadtxt('z_logvar_weights_TRAIN_kl5.txt',delimiter=',')
z_logvar_bias=np.loadtxt('z_logvar_bias_TRAIN_kl5.txt',delimiter=',')
decoder1_weights=np.loadtxt('decoder1_weights_TRAIN_kl5.txt',delimiter=',')
decoder1_bias=np.loadtxt('decoder1_bias_TRAIN_kl5.txt',delimiter=',')
output_weights=np.loadtxt('output_weights_TRAIN_kl5.txt',delimiter=',')
output_bias=np.loadtxt('output_bias_TRAIN_kl5.txt',delimiter=',')


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
     
  def multi_transition(self,action):
      action_vec=np.array([0,0,0,0,0,0]).reshape(1,6)
      action_vec[0,action]=1
      action_array=np.array([action_vec.reshape(-1)]*self.state.shape[0])
      input_array=np.concatenate([self.state,action_array],axis=1)
      self.state=self.predict(input_array)
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
     
        
#Create LSTM environment
class lstm_env:
    def __init__(self):
        self.model=tf.keras.Sequential([
            tf.keras.layers.LSTM(units=10,activation='tanh',input_shape=[47,20],return_sequences=True,use_bias=False),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(14,activation='linear',use_bias=False))                
            ])
        self.model.load_weights("lstm_weights_TRAIN.h5")
        self.emis_sds=np.loadtxt("lstm_sds_TRAIN.txt",delimiter=",")
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
gan_hidden_weights=np.loadtxt('GAN Weights Folder/gan_hidden_weights_TRAIN.txt',delimiter=',')
gan_hidden_bias=np.loadtxt('GAN Weights Folder/gan_hidden_bias_TRAIN.txt',delimiter=',')
gan_output_weights=np.loadtxt('GAN Weights Folder/gan_output_weights_TRAIN.txt',delimiter=',')
gan_output_bias=np.loadtxt('GAN Weights Folder/gan_output_bias_TRAIN.txt',delimiter=',')

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


#CVAE
cvae_hidden_weights=np.loadtxt('cvae_decoder1_weights_TRAIN_25_LONG.txt',delimiter=',')
cvae_hidden_bias=np.loadtxt('cvae_decoder1_bias_TRAIN_25_LONG.txt',delimiter=',')
cvae_output_weights=np.loadtxt('cvae_output_weights_TRAIN_25_LONG.txt',delimiter=',')
cvae_output_bias=np.loadtxt('cvae_output_bias_TRAIN_25_LONG.txt',delimiter=',')

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
    
    def multi_transition(self,action):
        action_vec=np.array([0,0,0,0,0,0]).reshape(1,6)
        action_vec[0,action]=1
        action_array=np.array([action_vec.reshape(-1)]*self.state.shape[0])
        input_array=np.concatenate([self.state,action_array],axis=1)
        self.state=self.model.predict(input_array)
        return(self.state)


#Define simulated trajectory function
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
            
    #if environment=="vae":
    #    final_aptt=[]
    #    trajectories=[]
    #    for i in range(ntrials):
    #        env=vae_env()
    #        env.state=first_state.reshape(1,14)
    #        aptt_meas=[first_state[0]]
    #        for step in range(len(action_list)-1):
    #            aptt_meas.append(env.transition(action=action_list[step])[0])
    #        final_aptt.append((aptt_meas[-1]*z_std[0])+z_means[0])
    #        aptt_meas=[aptt*z_std[0]+z_means[0] for aptt in aptt_meas]
    #        trajectories.append(aptt_meas)
    
    if environment=="vae":
        trajectories=[]
        first_state=np.array([first_state]*ntrials)
        
        env=vae_env()
        env.state=first_state
        aptt_meas=env.state[:,0].reshape((ntrials,1))
        
        for step in range(len(action_list)-1):
            env.multi_transition(action=action_list[step])
            aptt_meas=np.concatenate([aptt_meas,env.state[:,0].reshape((ntrials,1))],axis=1)
        
        final_aptt=[aptt_meas[i,-1]*z_std[0]+z_means[0] for i in range(ntrials)]
        aptt_meas=aptt_meas*z_std[0]+z_means[0]
        trajectories=[aptt_meas[i,:] for i in range(ntrials)]

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
    
    #if environment=="cvae":
    #    final_aptt=[]
    #    trajectories=[]
    #    for i in range(ntrials):
    #        env=cvae_env()
    #        env.state=first_state.reshape(1,14)
    #        aptt_meas=[first_state[0]]
    #        for step in range(len(action_list)-1):
    #            aptt_meas.append(env.transition(action=action_list[step])[0])
    #        final_aptt.append((aptt_meas[-1]*z_std[0])+z_means[0])
    #        aptt_meas=[aptt*z_std[0]+z_means[0] for aptt in aptt_meas]
    #        trajectories.append(aptt_meas)  
    
    if environment=="cvae":
        print('its working')
        trajectories=[]
        first_state=np.array([first_state]*ntrials)
        
        env=cvae_env()
        env.state=first_state
        aptt_meas=env.state[:,0].reshape((ntrials,1))
        
        for step in range(len(action_list)-1):
            env.multi_transition(action=action_list[step])
            aptt_meas=np.concatenate([aptt_meas,env.state[:,0].reshape((ntrials,1))],axis=1)
        
        final_aptt=[aptt_meas[i,-1]*z_std[0]+z_means[0] for i in range(ntrials)]
        aptt_meas=aptt_meas*z_std[0]+z_means[0]
        trajectories=[aptt_meas[i,:] for i in range(ntrials)]
    
    mse=np.mean((np.array(final_aptt)-real_aptt)**2)
    if ret=="mse":
        return mse
    if ret=="aptt":
        return final_aptt
    if ret=='trajectory':
        return trajectories    


#STANDARDIZED SET OF ALTERNATE TRAJECTORIES (100 per participant, same length as their trajectory)
import scipy.stats
val_ids=np.loadtxt("val_ids.txt",delimiter=",")
fulldata=pd.read_csv("Nemani Patient Data Cleaned Imputed.csv")
fulldata=fulldata[fulldata.INDEX.isin(val_ids)]
ind=np.unique(fulldata.INDEX)

#lengths=[len(list(fulldata[fulldata.INDEX==i]["aPTT"])) for i in ind]
#incl_ids=np.array(ind)[np.array(lengths)>=4]
#val_ids=incl_ids;ind=incl_ids;fulldata=fulldata[fulldata.INDEX.isin(incl_ids)]
#incl_index=np.where(np.array(lengths)>=4)

for i in range(len(ind)):
    print(i/len(ind))
    #vae_seq=np.array(mse_eval(ind[i],environment="vae",ntrials=100,ret="trajectory"))
    #hmm_seq=np.array(mse_eval(ind[i],environment="hmm",ntrials=100,ret="trajectory"))
    #lstm_seq=np.array(mse_eval(ind[i],environment="lstm",ntrials=100,ret="trajectory"))
    #gan_seq=np.array(mse_eval(ind[i],environment="gan",ntrials=100,ret="trajectory"))
    cvae_seq=np.array(mse_eval(ind[i],environment="cvae",ntrials=100,ret="trajectory"))
    
    #np.savetxt("patient"+str(i)+"train_kl5.txt",vae_seq,delimiter=",")
    #np.savetxt("patient"+str(i)+"train.txt",hmm_seq,delimiter=",")
    #np.savetxt("patient"+str(i)+"train.txt",lstm_seq,delimiter=",")
    #np.savetxt("patient"+str(i)+"train.txt",gan_seq,delimiter=",")
    np.savetxt("patient"+str(i)+"train_25_LONG.txt",cvae_seq,delimiter=",")


#OFFICIAL TRAJECTORY CHARACTERISTIC RESULTS
val_ids=np.loadtxt("val_ids.txt",delimiter=",")
fulldata=pd.read_csv("Nemani Patient Data Cleaned Imputed.csv")
fulldata=fulldata[fulldata.INDEX.isin(val_ids)]
ind=np.unique(fulldata.INDEX)


    #Check which KL weight VAE and CVAE are set to
seqlist_vae=[np.loadtxt("patient"+str(i)+"train.txt",delimiter=",") for i in range(len(ind))]
seqlist_hmm=[np.loadtxt("patient"+str(i)+"train.txt",delimiter=",") for i in range(len(ind))]
seqlist_lstm=[np.loadtxt("patient"+str(i)+"train.txt",delimiter=",") for i in range(len(ind))]
seqlist_gan=[np.loadtxt("patient"+str(i)+"train.txt",delimiter=",") for i in range(len(ind))]
seqlist_cvae=[np.loadtxt("patient"+str(i)+"train_25.txt",delimiter=",") for i in range(len(ind))]


#Final aPTT mae
final_real=[list(fulldata[fulldata.INDEX==i]["aPTT"])[-1] for i in ind]
full_real=[list(fulldata[fulldata.INDEX==i]["aPTT"]) for i in ind]
lengths=[len(list(fulldata[fulldata.INDEX==i]["aPTT"])) for i in ind]

vae_mae=[np.mean(np.absolute(seqlist_vae[i]-np.array(full_real[i]))) for i in range(len(val_ids))]
hmm_mae=[np.mean(np.absolute(seqlist_hmm[i]-np.array(full_real[i]))) for i in range(len(val_ids))]
lstm_mae=[np.mean(np.absolute(seqlist_lstm[i]-np.array(full_real[i]))) for i in range(len(val_ids))]
gan_mae=[np.mean(np.absolute(seqlist_gan[i]-np.array(full_real[i]))) for i in range(len(val_ids))]
cvae_mae=[np.mean(np.absolute(seqlist_cvae[i]-np.array(full_real[i]))) for i in range(len(val_ids))]

np.mean(vae_mae)
np.mean(hmm_mae)
np.mean(lstm_mae)
np.mean(gan_mae)
np.mean(cvae_mae)

vae_se=1.96*np.std(vae_mae)/np.sqrt(len(vae_mae))
hmm_se=1.96*np.std(hmm_mae)/np.sqrt(len(hmm_mae))
lstm_se=1.96*np.std(lstm_mae)/np.sqrt(len(lstm_mae))
gan_se=1.96*np.std(gan_mae)/np.sqrt(len(gan_mae))
cvae_se=1.96*np.std(cvae_mae)/np.sqrt(len(cvae_mae))

plt.rc('font', size=16)
fig= plt.figure(figsize=(6,6))
plt.errorbar(x=['CGAN','tVAE','CVAE','POMDP','LSTM'],
             y=[np.mean(gan_mae),np.mean(vae_mae),np.mean(cvae_mae),np.mean(hmm_mae),np.mean(lstm_mae)],
             yerr=[gan_se,vae_se,cvae_se,hmm_se,lstm_se],fmt='o',color='black',capsize=12)

#Average aPTT (tVAE is closer)
vae_avg=[np.mean(seq) for seq in seqlist_vae]
hmm_avg=[np.mean(seq) for seq in seqlist_hmm]
lstm_avg=[np.mean(seq) for seq in seqlist_lstm]
gan_avg=[np.mean(seq) for seq in seqlist_gan]
cvae_avg=[np.mean(seq) for seq in seqlist_cvae]

np.mean(vae_avg)
np.mean(hmm_avg)
np.mean(lstm_avg)
np.mean(gan_avg)
np.mean(cvae_avg)

vae_se=1.96*np.std(vae_avg)/np.sqrt(len(vae_avg))
hmm_se=1.96*np.std(vae_avg)/np.sqrt(len(hmm_avg))
lstm_se=1.96*np.std(vae_avg)/np.sqrt(len(lstm_avg))
gan_se=1.96*np.std(vae_avg)/np.sqrt(len(gan_avg))
cvae_se=1.96*np.std(vae_avg)/np.sqrt(len(cvae_avg))

np.mean([np.mean(full) for full in full_real])

plt.rc('font', size=16)
fig= plt.figure(figsize=(6,6))
plt.errorbar(x=['CGAN','tVAE','CVAE','POMDP','LSTM'],
             y=[np.mean(gan_avg),np.mean(vae_avg),np.mean(cvae_avg),np.mean(hmm_avg),np.mean(lstm_avg)],
             yerr=[gan_se,vae_se,cvae_se,hmm_se,lstm_se],fmt='o',color='black',capsize=12)
plt.hlines(61.12,xmin=0,xmax=4,linestyles='dashed')

#Average absolute difference. Can use median because positive skew
vae_diff=[np.mean(np.absolute(np.diff(seq,axis=1))) for seq in seqlist_vae]
hmm_diff=[np.mean(np.absolute(np.diff(seq,axis=1))) for seq in seqlist_hmm]
lstm_diff=[np.mean(np.absolute(np.diff(seq,axis=1))) for seq in seqlist_lstm]
gan_diff=[np.mean(np.absolute(np.diff(seq,axis=1))) for seq in seqlist_gan]
cvae_diff=[np.mean(np.absolute(np.diff(seq,axis=1))) for seq in seqlist_cvae]

np.median(vae_diff)
np.median(hmm_diff)
np.median(lstm_diff)
np.median(gan_diff)
np.median(cvae_diff)

    #Real difference
real_seq=[list(fulldata[fulldata.INDEX==i]["aPTT"]) for i in val_ids]
absdiff=[np.mean(np.absolute(np.diff(seq))) for seq in real_seq]
np.median(absdiff)

#Average absolute percentage change
differences=np.concatenate([np.absolute(np.diff(seq)) for seq in real_seq])
prev_aptt=np.concatenate([seq[0:-1] for seq in real_seq])
perc_diff=differences/prev_aptt

vae_diff=np.concatenate([np.absolute(np.diff(seq,axis=1)).reshape(-1) for seq in seqlist_vae])
hmm_diff=np.concatenate([np.absolute(np.diff(seq,axis=1)).reshape(-1) for seq in seqlist_hmm])
lstm_diff=np.concatenate([np.absolute(np.diff(seq,axis=1)).reshape(-1) for seq in seqlist_lstm])
gan_diff=np.concatenate([np.absolute(np.diff(seq,axis=1)).reshape(-1) for seq in seqlist_gan])
cvae_diff=np.concatenate([np.absolute(np.diff(seq,axis=1)).reshape(-1) for seq in seqlist_cvae])

vae_prev=np.concatenate([seq[:,0:-1].reshape(-1) for seq in seqlist_vae])
hmm_prev=np.concatenate([seq[:,0:-1].reshape(-1) for seq in seqlist_hmm])
lstm_prev=np.concatenate([seq[:,0:-1].reshape(-1) for seq in seqlist_lstm])
gan_prev=np.concatenate([seq[:,0:-1].reshape(-1) for seq in seqlist_gan])
cvae_prev=np.concatenate([seq[:,0:-1].reshape(-1) for seq in seqlist_cvae])

vae_perc=(vae_diff/vae_prev)[vae_prev>0]
hmm_perc=(hmm_diff/hmm_prev)[hmm_prev>0]
lstm_perc=(lstm_diff/lstm_prev)[lstm_prev>0]
gan_perc=(gan_diff/gan_prev)[gan_prev>0]
cvae_perc=(cvae_diff/cvae_prev)[cvae_prev>0]

import seaborn as sns
from matplotlib.ticker import PercentFormatter
plt.rc('font', size=12)

data_dict={'Mean Absolute % Change':np.concatenate([perc_diff,vae_perc,hmm_perc,lstm_perc,gan_perc,cvae_perc]),'Model': ['Data']*len(perc_diff)+['tVAE']*len(vae_perc)+['POMDP']*len(hmm_perc)+['LSTM']*len(lstm_perc)+['CGAN']*len(gan_perc)+['CVAE']*len(cvae_perc)}
boxdata=pd.DataFrame.from_dict(data_dict)
plot=sns.boxplot(x='Model',y="Mean Absolute % Change",data=boxdata,color='gray',showfliers=False)
plt.ylim([0,1])
plot.yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig("percentage_change.eps",dpi=300,format="eps")

np.mean(perc_diff)
np.mean(vae_perc)
np.mean(hmm_perc)
np.mean(lstm_perc)
np.mean(gan_perc)
np.mean(cvae_perc)

np.mean(perc_diff>.1)
np.mean(vae_perc>.1)
np.mean(hmm_perc>.1)
np.mean(lstm_perc>.1)
np.mean(gan_perc>.1)
np.mean(cvae_perc>.1)

#Heteroskedasticity
qs=[0,60,100,1000]

vae_prev=vae_prev[vae_prev>0]
hmm_prev=hmm_prev[hmm_prev>0]
lstm_prev=lstm_prev[lstm_prev>0]
gan_prev=gan_prev[gan_prev>0]
cvae_prev=cvae_prev[cvae_prev>0]

vae_res=[np.mean(vae_perc[(vae_prev>qs[i]) & (vae_prev<qs[i+1])]) for i in range(len(qs)-1)]
hmm_res=[np.mean(hmm_perc[(hmm_prev>qs[i]) & (hmm_prev<qs[i+1])]) for i in range(len(qs)-1)]
lstm_res=[np.mean(lstm_perc[(lstm_prev>qs[i]) & (lstm_prev<qs[i+1])]) for i in range(len(qs)-1)]
gan_res=[np.mean(gan_perc[(gan_prev>qs[i]) & (gan_prev<qs[i+1])]) for i in range(len(qs)-1)]
cvae_res=[np.mean(cvae_perc[(cvae_prev>qs[i]) & (cvae_prev<qs[i+1])]) for i in range(len(qs)-1)]
real_res=[np.mean(perc_diff[(prev_aptt>qs[i]) & (prev_aptt<qs[i+1])]) for i in range(len(qs)-1)]


fig, ax=plt.subplots()
ax.plot(real_res,color="black",label="Data")
ax.plot(vae_res,color=".5",label="tVAE")
ax.plot(hmm_res,color="black",linestyle=":",label="HMM")
ax.plot(lstm_res,color=".5",linestyle=":",label="LSTM")
ax.plot(gan_res,color="black",linestyle="-.",label="CGAN")
ax.plot(cvae_res,color=".5",linestyle="-.",label="CVAE")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.xticks([0,1,2],labels=["aPTT<60","60<aPTT<100","aPTT>100"])
plt.xlabel("Clinical aPTT Range")
plt.ylabel("Mean Absolute % Change")
plt.tight_layout()
plt.legend()
plt.savefig("perc_clinical_window.eps",dpi=300,format="eps")


#Autocorrelations. CVAE and VAE are closest
aptt_first=[list(fulldata[fulldata.INDEX==i]["aPTT"])[0:-1] for i in val_ids]
aptt_next=[list(fulldata[fulldata.INDEX==i]["aPTT"])[1:] for i in val_ids]
aptt_first=np.concatenate(aptt_first)
aptt_next=np.concatenate(aptt_next)

np.corrcoef(aptt_first,aptt_next)

vae_first=[list(seqlist_vae[i][:,0:-1].reshape(-1)) for i in range(len(val_ids))]
vae_next=[list(seqlist_vae[i][:,1:].reshape(-1)) for i in range(len(val_ids))]
vae_first=np.concatenate(vae_first)
vae_next=np.concatenate(vae_next)
np.corrcoef(vae_first,vae_next)

hmm_first=[list(seqlist_hmm[i][:,0:-1].reshape(-1)) for i in range(len(val_ids))]
hmm_next=[list(seqlist_hmm[i][:,1:].reshape(-1)) for i in range(len(val_ids))]
hmm_first=np.concatenate(hmm_first)
hmm_next=np.concatenate(hmm_next)
np.corrcoef(hmm_first,hmm_next)

lstm_first=[list(seqlist_lstm[i][:,0:-1].reshape(-1)) for i in range(len(val_ids))]
lstm_next=[list(seqlist_lstm[i][:,1:].reshape(-1)) for i in range(len(val_ids))]
lstm_first=np.concatenate(lstm_first)
lstm_next=np.concatenate(lstm_next)
np.corrcoef(lstm_first,lstm_next)

gan_first=[list(seqlist_gan[i][:,0:-1].reshape(-1)) for i in range(len(val_ids))]
gan_next=[list(seqlist_gan[i][:,1:].reshape(-1)) for i in range(len(val_ids))]
gan_first=np.concatenate(gan_first)
gan_next=np.concatenate(gan_next)
np.corrcoef(gan_first,gan_next)

cvae_first=[list(seqlist_cvae[i][:,0:-1].reshape(-1)) for i in range(len(val_ids))]
cvae_next=[list(seqlist_cvae[i][:,1:].reshape(-1)) for i in range(len(val_ids))]
cvae_first=np.concatenate(cvae_first)
cvae_next=np.concatenate(cvae_next)
np.corrcoef(cvae_first,cvae_next)

#Instances of negative correlations
vae_cor=[]
hmm_cor=[]
lstm_cor=[]
gan_cor=[]
cvae_cor=[]

counter=-1
for i in range(len(ind)):
    print(i)
    real=fulldata[fulldata.INDEX==ind[i]]["aPTT"]
    
    vae_cor.append(np.mean([np.corrcoef(seqlist_vae[i][row,:],real)[0,1] for row in range(100)]))
    hmm_cor.append(np.mean([np.corrcoef(seqlist_hmm[i][row,:],real)[0,1] for row in range(100)]))
    lstm_cor.append(np.mean([np.corrcoef(seqlist_gan[i][row,:],real)[0,1] for row in range(100)]))
    gan_cor.append(np.mean([np.corrcoef(seqlist_lstm[i][row,:],real)[0,1] for row in range(100)]))
    cvae_cor.append(np.mean([np.corrcoef(seqlist_cvae[i][row,:],real)[0,1] for row in range(100)]))


np.mean(np.array(vae_cor)<0) #.14838
np.mean(np.array(hmm_cor)<0) #.16774
np.mean(np.array(lstm_cor)<0) #.15806
np.mean(np.array(gan_cor)<0) #.15484
np.mean(np.array(cvae_cor)<0) #.23871

#PLOTS
#index=[256,354,990,1372,1478][4]
index=110

plt.plot(seqlist_vae[index][0:10,:].T,'-o',c="0")
plt.ylim([20,140])
plt.show()

plt.plot(seqlist_hmm[index][0:10,:].T,'-o',c="0")
plt.ylim([20,140])
plt.show()

plt.plot(seqlist_lstm[index][0:10,:].T,'-o',c="0")
plt.ylim([20,140])
plt.show()

plt.plot(seqlist_gan[index][0:10,:].T,'-o',c="0")
plt.ylim([20,140])
plt.show()

plt.plot(seqlist_cvae[index][0:10,:].T,'-o',c="0")
plt.ylim([20,140])
plt.show()

plt.plot(fulldata.aPTT[fulldata.INDEX==ind[index]],'-o',c="0")
plt.ylim([20,140])
plt.show()

#Look at slopes
real_slope=[]
for i in ind:
   y=fulldata[fulldata.INDEX==i]['aPTT']
   x=np.array(range(len(y)))
   real_slope.append(np.corrcoef(x,y)[0,1]*np.std(y)/np.std(x))

vae_slope=[]
for seq in seqlist_vae:
    y=seq
    x=np.array(range(seq.shape[1]))
    
    rs=[np.corrcoef(y[i,:],x)[0,1] for i in range(y.shape[0])]
    sy=[np.std(y[i,:]) for i in range(y.shape[0])]
    vae_slope.append(np.array(rs)*np.array(sy)/np.std(x))

cvae_slope=[]
for seq in seqlist_cvae:
    y=seq
    x=np.array(range(seq.shape[1]))
    
    rs=[np.corrcoef(y[i,:],x)[0,1] for i in range(y.shape[0])]
    sy=[np.std(y[i,:]) for i in range(y.shape[0])]
    cvae_slope.append(np.array(rs)*np.array(sy)/np.std(x))


    

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




