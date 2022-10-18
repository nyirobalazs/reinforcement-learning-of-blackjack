import numpy as np
from dependencies import *

def episode(player, decks, training = False):
    try:
        hand_sum = 0
        hand_score = 0
        episode_score = 0
        hands_played = 0
        hand_card_history = []

        number_of_decks = int(decks.cards.size/52)
        if decks.restack:
            number_of_decks = 0

        trajectory = []
        previous_state = get_state(0, number_of_decks, decks,'void')
        first_reward = 0
        trajectory.append([previous_state, first_reward])

        print_hand_header(hands_played)
        while True:
            if len(hand_card_history) == 0:
                action = 'h'
            else:
                action = player.select_next_action()

            trajectory[-1].append(action)
            
            if action == 'h':
                card = decks.draw_card()
                hand_card_history.append(card)
                hand_sum = get_hand_sum(hand_card_history)
                print_card_deal(card, hand_sum)

                if hand_sum >21:
                    end_hand = True
                else:
                    end_hand = False
            elif action == 's':
                end_hand = True

            state = get_state(hand_sum, number_of_decks, decks, action)
            reward = get_reward(state, action, previous_state)
            trajectory.append([state, reward])
            previous_state = state

            # The player updates its state according to the last step
            player.update_state(state)

            if end_hand:
                hand_score = get_hand_score(hand_sum)
                episode_score += hand_score
                print_end_of_hand(hand_score,episode_score)
                hand_card_history = []
                end_hand = False

                if training:
                    # Check if deck is infinite, if so, episode ends after each
                    # hand so that the agent can learn
                    if decks.restack == True:
                        raise

                hands_played += 1
                hand_sum = 0
                print_hand_header(hands_played)
    except:
        hand_score = get_hand_score(hand_sum)
        episode_score += hand_score
        
        flag_to_final_state = -1
        # Assign the last state and reward to the
        # agent before finishing the episode
        if len(trajectory[-1]) == 3: # Deck ran out of cards
            if hand_score > 0:
                state = get_state(flag_to_final_state, number_of_decks, decks, 's')
                reward = get_reward(state, 's', previous_state)
                trajectory.append([state, reward])
        else: # len(trajectory) == 2
            state = get_state(flag_to_final_state, number_of_decks, decks, 's')
            trajectory[-1][0] = state

        if training: 
            return trajectory
        else:
            print("You are out of cards! Final score: %s"%(episode_score))
            return episode_score

def get_hand_sum(hand_card_history):
    new_hand_sum = 0
    num_of_aces = 0

    for card in hand_card_history:
        if card.name == 'A':
            # Count all aces in hand; at the end if possible, add 10 as
            # many times as allowed
            num_of_aces += 1

        new_hand_sum += card.value
    
    for j in range(num_of_aces):
        if new_hand_sum <= 11:
            new_hand_sum += 10
    
    return new_hand_sum

def print_card_deal(card, hand_sum):
    if card.name == 'A':
        print("%s\t %s/11\t%s"%(card.name, card.value, hand_sum))
    else:
        print("%s\t %s\t%s"%(card.name, card.value, hand_sum))

def get_state(hand_sum, number_of_decks, decks, action):
    state = -1*np.ones((5,))
    
    if hand_sum == -1:
        state[0] = -1
    else:
        # Go back to state 0 if previous action was 'stick' or if last hand was failed
        if action == 's' or hand_sum > 21:
            state[0] = 0
        else:
            state[0] = hand_sum

        state[1] = number_of_decks
    
        (P_low, P_med, P_high) = count_cards(decks)
        state[2] = P_low
        state[3] = P_med
        state[4] = P_high
    return state

def count_cards(decks):
    '''The probability of the next card being of any given value is known to the
    player. In the real world, if the player knows how many decks there are, and
    keeps track of the cards that have been dealt, he can iteratively keep track
    of the likelihood of the next card. In this case we simplify the
    calculations by just counting all the cards in the decks and returning the
    likelihood of each value being the next card dealt.
    '''
    number_of_cards = decks.cards.size
    prob_df = pd.DataFrame({"name":["2","3","4","5","6","7","8","9","10","J","Q","K","A"],\
                            "count":np.zeros((13,)),\
                            "likelihood":np.zeros((13,))})
    for card in decks.cards:
        prob_df.loc[prob_df["name"] == card.name, "count"] += 1

    prob_df.loc[:,"likelihood"] = prob_df.loc[:,"count"]/number_of_cards

    likelihoods = prob_df.loc[:,"likelihood"].values
    P_low = np.sum(likelihoods[:5]) #Probability of low card (2-6)
    P_med = np.sum(likelihoods[5:8]) # Probability of medium card (7-9)
    P_high = np.sum(likelihoods[8:]) # Probability of high card (10-Ace)
    return (P_low, P_med, P_high)

def get_reward(state, previous_action, previous_state):
    hand_sum = previous_state[0]

    # Reward equal the hand score, only if it's the end of the hand
    if previous_action == 's':
        reward = get_hand_score(hand_sum) 
    else:
        reward = 0
    return reward

def get_hand_score(hand_sum):
    if (hand_sum <= 21):
        hand_score = hand_sum**2
    else:
        hand_score = 0
    return hand_score

def print_end_of_hand(hand_score,episode_score):
    print("Hand score: %s | Episode score: %s"%(hand_score,episode_score))

def print_hand_header(hands_played):
    print("\n")
    print("________________________")
    print("Hand # %s:"%(hands_played))
    print("Card\tValue\tHand sum")
    print("------------------------")


if __name__ == '__main__':
    player = Player()
    decks = Decks(1)
    _ = episode(player,decks)
