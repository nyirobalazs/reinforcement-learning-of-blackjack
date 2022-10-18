import time
import numpy as np
import random
import pandas as pd
import math

class Card:

    def __init__(self,name,suit):
        self.name = name
        self.value = self.__get_value(name)
        self.suit = suit

    def __get_value(self,name):
        try:
            value = int(name)
        except: # name cannot be converted to int, then it's A,J,Q or K
            if name == 'A':
                value = 1
            else:
                value = 10
        return value


class Decks:
    
    def __init__(self,number_of_decks = 0):
        
        if number_of_decks == 0: # Create only 1 deck, but re-stack it after each action
            number_of_decks = 1
            self.restack = True
        else:
            self.restack = False

        suits = ["Hearts", "Clubs", "Diamonds", "Spades"]
        non_numeric_cards = ["J", "Q", "K", "A"]

        self.cards = np.empty((number_of_decks,len(suits),13),dtype = Card)

        for i in range(number_of_decks):
            for k in range(len(suits)):
                for j in range(9): # Cards with numeric name
                    self.cards[i,k,j] = Card(str(j+2),suits[k])
                for j in range(9,13): # Cards with non-numeric name
                    self.cards[i,k,j] = Card(non_numeric_cards[j-9],suits[k])

         # To end up with a one dimensional array of cards
        self.cards = self.cards.flatten()

    def draw_card(self):
        card_index = random.choice(range(self.cards.size))
        card = self.cards[card_index]

        if self.restack == False:
            self.cards = np.delete(self.cards,card_index)

        return card

class Player:
    def __init__(self):
        self.actions = ['s','h']
    
    def select_next_action(self):
        action = input('(h/s):\t')
        return action

    def update_state(self, new_state):
        self.state = new_state

class Agent(Player):
    def __init__(self, sleep = True, epsilon = 0):
        super().__init__()
        
        # Determine if the agents sleeps before it chooses an action, when
        # training, it is set False by the constructor
        self.sleep = sleep       
        self.epsilon = epsilon
        self.W = self.__get_weights()

    def __get_weights(self):
        W = pd.read_csv("weights.csv",index_col=0)
        return W.values.flatten()
    
    def select_next_action(self):
        if self.sleep:
            time.sleep(1)
        
        # epsilon-greedy                                                     
        action_index, _ = self.__get_max_value_action(self.state)            
        # epsilon likelihood of exploring                                    
        if random.random() <= self.epsilon:                                  
            action_index = 1-action_index                                    
        return self.actions[action_index]

    def learn(self, trajectory):
        # get number of training sessions and add 1                              
        try:
            with open("episodes_trained.txt","r") as training_index_file: 
                last_training_index = int(training_index_file.read()) + 1
        except: # If the file is not found, just start as if it's the first training session
                last_training_index = 1

        # find last exploratory action                                           
        index_last_exploration = 0                                               
        for i in range(len(trajectory)-1):                                       
            state = trajectory[i][0]                                             
                                                                                 
            action = trajectory[i][2]                                            
            numeric_action = self.actions.index(action)                          
            _, max_value = self.__get_max_value_action(state)                    
                                                                                 
            if max_value != self.__get_state_action_value(state,numeric_action): 
                # Action was chosen non-greedily (exploration)                   
                index_last_exploration = i                                       
                                                                                 
        # SGD
        total_reward = 0                                                         
        alpha = 0.01
        nabla_loss = np.zeros(self.W.shape)
        T = len(trajectory) - index_last_exploration

        for i in range(len(trajectory)-2,index_last_exploration-1,-1):           
            state = trajectory[i][0]                                             
            action = trajectory[i][2]                                            
            numeric_action = self.actions.index(action)                          
            
            reward = trajectory[i+1][1]                                          
            total_reward += reward                                               
            
            nabla_Q = self.__build_state_action_vector(state, numeric_action)
            nabla_loss += -(1/T)*(total_reward - \
                          self.__get_state_action_value(state, numeric_action))*\
                          nabla_Q

        self.W = self.W - alpha * nabla_loss

        # add new W to history file                                              
        W_df = pd.DataFrame({"value":self.W})
        W_last_session = W_df.copy()                                           
        W_last_session.loc[:,"training index"] = last_training_index 

        with open("weights.csv","w") as W_file:
            W_file.write(W_df.to_csv())
        with open("weights_history.csv","a") as W_history_file:
            W_history_file.write(W_last_session.to_csv(header=False))
        with open("episodes_trained.txt","w") as training_index_file:
            training_index_file.write(str(last_training_index))

    def __get_max_value_action(self,state):                       
        max_value = -math.inf
        for i in range(len(self.actions)):
            value = self.__get_state_action_value(state,i)
            if value > max_value:
                max_value = value
                action_index = i
            elif value == max_value:
                # if both actions have same value, choose randomly                       
                action_index = np.random.choice([0,1])                          
        return action_index, max_value       

    def __get_state_action_value(self, state, action):
        state_action_vec = self.__build_state_action_vector(state,action)
        value = self.W @ state_action_vec
        value = value.item()
        return value

    def __build_state_action_vector(self,state,numeric_action):
        bias = 1
        hand_sum = state[0]
        number_of_decks = state[1]
        P_low = state[2]
        P_med = state[3]
        P_high = state[4]

        d = number_of_decks
        a = numeric_action
        dirac_d0 = (d==0)
        Heaviside_d1 = (d>0)

        h_vec = (1-a)*np.array([bias, hand_sum/21, (hand_sum/21)**2])
        a_vec = a *np.array([bias, hand_sum/21,(hand_sum/21)**2])
        P_high_vec = a*(P_high)*np.array([bias,(hand_sum/21)])
        
        state_vec_d0 = dirac_d0 * np.concatenate([h_vec,a_vec])
        state_vec_d_plus =  Heaviside_d1 * np.concatenate([h_vec,a_vec,P_high_vec])

        state_vec = np.concatenate([state_vec_d0, state_vec_d_plus])

        return state_vec
