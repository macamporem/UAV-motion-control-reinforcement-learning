



# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import sys
import pandas as pd
import time
import random

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"

import gym

import tensorflow as tf

reset_graph()

n_inputs = 6#4
n_hidden = 6#4
n_outputs = 8#1 (neutral, 2xroll,2xpitch,2xyaw, no-thrust)

learning_rate = 0.02#0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

#hidden = tf.layers.dense(X      , n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
#hidden = tf.layers.dense(X      , n_hidden, activation=tf.nn.elu,         kernel_initializer='random_uniform', bias_initializer='zeros')
#logits  = tf.layers.dense(hidden , n_outputs)
#logits  = tf.layers.dense(hidden , n_outputs,         kernel_initializer='random_uniform', bias_initializer='zeros')
logits  = tf.layers.dense(X , n_outputs, \
        kernel_initializer='random_uniform', bias_initializer='zeros')

#outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
outputs = logits
#p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
#action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

action = tf.argmax(logits,1)
    
#y = 1. - tf.to_float(action)
y = tf.squeeze(tf.one_hot(action,n_outputs))

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


def quad(controller_log):
    #Roll,Pitch
    try:
        df1 = pd.read_csv(controller_log+'rl-RollPitchControlState.txt')
    except:
        df1 = pd.read_csv(controller_log+'rl-RollPitchControlState-last.txt')
    if df1.shape[0]>=1:
        df1.to_csv(controller_log+'rl-RollPitchControlState-last.txt')
        df1 = df1.iloc[0,:]
    else:
        df1 = pd.read_csv(controller_log+'rl-RollPitchControlState-last.txt')
        df1 = df1.iloc[0,:]
        
    #Lateral position
    try:
        df2 = pd.read_csv(controller_log+'rl-LateralPositionControlState.txt')
    except:
        df2 = pd.read_csv(controller_log+'rl-LateralPositionControlState-last.txt')
    if df2.shape[0]>=1:
        df2.to_csv(controller_log+'rl-LateralPositionControlState-last.txt')
        df2 = df2.iloc[0,:]
    else:
        df2 = pd.read_csv(controller_log+'rl-LateralPositionControlState-last.txt')
        df2 = df2.iloc[0,:]
        
    #Altitude
    try:
        df3 = pd.read_csv(controller_log+'rl-AltitudeControlState.txt')
    except:
        df3 = pd.read_csv(controller_log+'rl-AltitudeControlState-last.txt')
    if df3.shape[0]>=1:
        df3.to_csv(controller_log+'rl-AltitudeControlState-last.txt')
        df3 = df3.iloc[0,:]
    else:
        df3 = pd.read_csv(controller_log+'rl-AltitudeControlState-last.txt')
        df3 = df3.iloc[0,:]

#     #Yaw
#     try:
#         df4 = pd.read_csv(controller_log+'rl-YawControlState.txt')
#     except:
#         df4 = pd.read_csv(controller_log+'rl-YawControlState-last.txt')
#     if df4.shape[0]>=1:
#         df4.to_csv(controller_log+'rl-YawControlState-last.txt')
#         df4 = df4.iloc[0,:]
#     else:
#         df4 = pd.read_csv(controller_log+'rl-YawControlState-last.txt')
#         df4 = df4.iloc[0,:]

    try:
        df5 = pd.read_csv(controller_log+'rl-BodyRateControlState.txt')
    except:
        df5 = pd.read_csv(controller_log+'rl-BodyRateControlState-last.txt')
    if df5.shape[0]>=1:
        df5.to_csv(controller_log+'rl-BodyRateControlState-last.txt')
        df5 = df5.iloc[0,:]
    else:
        df5 = pd.read_csv(controller_log+'rl-BodyRateControlState-last.txt')
        df5 = df5.iloc[0,:]

    quad_obs = np.array([                     df1[3],                     df1[7],                     df1[11],                     df5[3],                     df5[4],                     df5[5],                 ])

    quad_cost = np.sqrt(                    0*(df1[3]-1)**2 +                     0*(df1[7]-1)**2 +                     0*(df1[11]-1)**2 +                     df5[3]**2 +                     df5[4]**2 +                     df5[5]**2                 )
    
    quad_done = df3[2]>-0.1 #df3[2] is z
    
    quad_dfo = np.sqrt(                     (df2[0]-df2[2])**2 +                     (df2[4]-df2[6])**2 +                     (df3[0]-df3[2])**2
                )

    quad_rewards = -1*quad_cost

    return quad_obs, quad_rewards, quad_done, quad_dfo, df3[2]



mt = 1.18
mt_ = np.array([mt,mt,mt,mt])

controller_log = 'FCND-Controls-CPP-reinforcement-learning/config/log/'
np.savetxt(controller_log+'rl-commanded-rotor-speeds.txt', 
           np.reshape(mt_,[1,4]
                     ), delimiter=",")

action_val_dic = {
    0:np.array(mt_),
    1:np.array(mt_),
    2:np.array(mt_),
    3:np.array(mt_),
    4:np.array(mt_),
    5:np.array(mt_),
    6:np.array(mt_),
    7:np.array([0,0,0,0])}

MaxT = 2
MinT = 0#0.8

i=1
action_val_dic[i][0] = action_val_dic[i][0]*MaxT
action_val_dic[i][1] = action_val_dic[i][1]*MaxT
action_val_dic[i][2] = action_val_dic[i][2]*MinT
action_val_dic[i][3] = action_val_dic[i][3]*MinT

i=2
action_val_dic[i][0] = action_val_dic[i][0]*MinT
action_val_dic[i][1] = action_val_dic[i][1]*MinT
action_val_dic[i][2] = action_val_dic[i][2]*MaxT
action_val_dic[i][3] = action_val_dic[i][3]*MaxT

i=3
action_val_dic[i][0] = action_val_dic[i][0]*MaxT
action_val_dic[i][1] = action_val_dic[i][1]*MinT
action_val_dic[i][2] = action_val_dic[i][2]*MaxT
action_val_dic[i][3] = action_val_dic[i][3]*MinT

i=4
action_val_dic[i][0] = action_val_dic[i][0]*MinT
action_val_dic[i][1] = action_val_dic[i][1]*MaxT
action_val_dic[i][2] = action_val_dic[i][2]*MinT
action_val_dic[i][3] = action_val_dic[i][3]*MaxT

i=5
action_val_dic[i][0] = action_val_dic[i][0]*MaxT
action_val_dic[i][1] = action_val_dic[i][1]*MinT
action_val_dic[i][2] = action_val_dic[i][2]*MinT
action_val_dic[i][3] = action_val_dic[i][3]*MaxT

i=6
action_val_dic[i][0] = action_val_dic[i][0]*MinT
action_val_dic[i][1] = action_val_dic[i][1]*MaxT
action_val_dic[i][2] = action_val_dic[i][2]*MaxT
action_val_dic[i][3] = action_val_dic[i][3]*MinT

print(action_val_dic)

#speed test
if False:
    check = 0 # if too fast, will be different from 0
    avcheck = 0
    for k in range(1000):
        for i in range(4):
            avcheck += check
            print('\rIteration {} {} {} {}'.format(k,i,avcheck,check),end="")
            np.savetxt(controller_log+'rl-commanded-rotor-speeds.txt', 
                   np.reshape(action_val_dic[i],[1,4]
                             ), delimiter=",")
            time.sleep(0.2)
            check = 0
            try:
                df = pd.read_csv(controller_log+'rl-commanded-rotor-speeds-check.txt')
                df = df.iloc[0,:]
                for u in range(4):
                    check += (df[u]-action_val_dic[i][u])**2
                    check = round(check,1)
            except:
                check = 999


def set_thrust(t):
    np.savetxt(controller_log+'rl-commanded-rotor-speeds.txt', 
        np.reshape(t,[1,4]), delimiter=",")
    return None

def set_origin(threshold):
    dfo = 10
    while dfo > threshold:
        try:
            obs, _ , _ , dfo, z = quad(controller_log)
        except:
            continue#break
    return obs

def random_roll_f(lag):
    random_roll = int(round(random.uniform(0, 1),0)+1)
    #random_roll = int(round(random.uniform(0, 8)-0.5,0))
    random_t = random.uniform(0.5, 2)
    set_thrust(random_t*action_val_dic[random_roll])
    time.sleep(lag)
    return random_roll, random_t

def OLD_status_screen(iteration, game, step, random_roll, action_val, obs, reward):
    print("\nIteration: {}, Game:{}, step:{}, roll:{}, action:{}, pqr:{:.1f}|{:.1f}|{:.2f}, rewards:{:.1f}"         .format(iteration, game, step,random_roll,         action_val[0][0], obs[3], obs[4], obs[5], reward), end="    ")
    #print("\npqr {0:.1f} {1:.1f} {2:.1f}" \
    #    .format(obs[3],obs[4],obs[5]), end="    ")
    return None

def status_screen_header(my_list):
    row_format ="{:>12}" * (len(my_list) + 1)
    print(row_format.format("", *my_list))    

def status_screen(iteration, game, step, random_roll, action_val, obs, reward):
    my_list = [iteration, game, step, random_roll, action_val[0],             '{:.0f}|{:.0f}|{:.0f}'.format(obs[3], obs[4], obs[5]),             '{:.1f}'.format(reward)]
    row_format ="{:>12}" * (len(my_list) + 1)
    print(row_format.format("", *my_list))    
    
def status_screen_(outputs):
    outputs = outputs[0]
    print("\nOutputs {0:.1f} {1:.1f} {2:.1f} {3:.1f} {4:.1f} {5:.1f} {6:.1f} {7:.1f}"         .format(outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7]), end="    ")
    return None


#system('rm ./saved_model/my_policy_net*')

n_iterations = 100#250
n_games_per_update = 100
n_max_steps = 1#1000
save_iterations = 1#10
discount_rate = 0.8#0.95
lag = 1.5#0.25
best_reward = - 999

with tf.Session() as sess:
    saver.restore(sess, "./saved_model/my_policy_net_pg.ckpt")

    set_thrust(mt_)

    for step in list([1,1,2,2]):
        print('\n')
        obs = set_origin(0.005)

        #random_roll = int(round(random.uniform(0, 1),0)+1)
        #random_t = random.uniform(1, 10)
        random_roll = step
        random_t = 1
        
        set_thrust(random_t*action_val_dic[random_roll])
        time.sleep(lag)
        #read state again
        obs, _ , _ , dfo, z = quad(controller_log)            
        
        #action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
        action_val, gradients_val, y_val, out_val, log_val =             sess.run([action, gradients, y, outputs, logits],             feed_dict={X: obs.reshape(1, n_inputs)})
        set_thrust(action_val_dic[action_val[0]])
        
        time.sleep(lag)
        obs, reward, done, _ , z = quad(controller_log)
        status_screen_header(['Iterat.','Game','step','roll','action','pqr','rewards'])
        status_screen(0, 0, 0, random_roll, action_val, obs, reward)
        status_screen_(log_val)
        status_screen_(out_val)
        #print("\nOutput: {}, Game:{}, step:{}, action:{}, rewards{:.0f}" \
        #        .format(iteration, game, step,action_val[0][0], reward), end="    ")

        #if done:
        #    break

