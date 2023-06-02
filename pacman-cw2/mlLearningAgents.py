# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        # The current state of the game
        self.state = state

        # The Q value of the state
        self.q_value = util.Counter()

        # The number of times we have visited this state
        self.visits = util.Counter()

        # The score of the state
        # self.score = state.getScore()


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # The Q value of the state
        self.q_value = util.Counter()
        # The number of times we have visited this state
        self.visits = util.Counter()
        # Number of times that a state-action pair is forced to be picked for exploration
        self.NE = 5
        # The optimistic reward for exploration
        self.R_PLUS = 2
        # The previous action stored
        self.lastAction = None
        # The previous state stored
        self.lastState = None
        # The table that stores the count of every state
        self.counts = {}
        # Reward of each state gained from ghosts, food and capsules
        self.STATE_REWARD = 0
        # Ghost reward
        self.GHOST_REWARD = -25
        # Food reward
        self.FOOD_REWARD = 12
        # Capsule reward
        self.CAPSULE_REWARD = 8

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Restore the history state and action for learning
    def setLastAction(self, action):
        self.lastAction = action

    def setLastState(self, state):
        self.lastState = state

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        reward = endState.getScore() - startState.getScore()
        return reward

    # Distance between pacman and object
    def objDistance(self, pacman_pos, object_pos, state: GameState) -> float:
        walls = state.getWalls()
        # Input the coordinates of two points (x1,y1), (x2,y2) and returns the shortest distance between these points
        # (Calculate the blocking effect of walls)
        queue = util.Queue()
        visited = []
        queue.push((pacman_pos, 0))
        while (not queue.isEmpty()):
            (point, distance) = queue.pop()
            if point not in visited:
                visited.append(point)
                if point == object_pos:
                    return distance
                # NSEW
                north = (point[0], point[1] + 1)
                south = (point[0], point[1] - 1)
                east = (point[0] + 1, point[1])
                west = (point[0] - 1, point[1])
                surroundings = [north, south, west, east]
                for point in surroundings:
                    # not yet visited and not walls
                    if point not in visited and point not in walls:
                        queue.push((point, distance + 1))
        return False

    # Update reward for specific state
    def updateReward(self, state: GameState) -> float:
        pacmanPosition = state.getPacmanPosition()
        ghostPosition = state.getGhostPositions()
        foodPosition = state.getFood()
        capsulePosition = state.getCapsules()
        reward = 0
        # Reward for food
        if pacmanPosition in foodPosition and pacmanPosition not in ghostPosition:
            if len(foodPosition) == 1:  # Increase the reward of the last food
                reward += self.FOOD_REWARD + 15
            else:
                reward += self.FOOD_REWARD
        # Reward for capsule
        elif pacmanPosition in capsulePosition and pacmanPosition not in ghostPosition:
            reward += self.CAPSULE_REWARD
        # Reward for ghost
        elif pacmanPosition in ghostPosition:
            for singleGhostPosition in ghostPosition:
                if singleGhostPosition == pacmanPosition:
                    reward += self.GHOST_REWARD
        # Radiate rewards to ghosts' surroundings
        for singleGhostPosition in ghostPosition:
            ghost_distance = self.objDistance(pacmanPosition, singleGhostPosition, state)
            if ghost_distance < 2:
                reward += self.GHOST_REWARD / (1+ghost_distance)
        # Update copy of reward
        self.STATE_REWARD = reward
        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.q_value[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        q_list = []
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        for action in legal:
            q = self.getQValue(state,action)
            q_list.append(q)
        if len(q_list) ==0:
            return 0
        return max(q_list)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        self.q_value[(state, action)] = (1 - self.alpha) * self.q_value[(state, action)] + self.alpha * (
                    reward + self.gamma * self.maxQValue(nextState))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # if haven's been to this state, then initial count to 1
        if (state, action) not in self.counts.keys():
            self.counts[(state, action)] = 1
        # else increase the count by 1
        else:
            self.counts[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        # if haven's been to this state, then initial count to 1
        if (state, action) not in self.counts.keys():
            self.counts[(state, action)] = 1
        return self.counts[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if counts < self.NE:
            return self.R_PLUS
        else:
            return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Do the Q learning upgrade
        if self.lastAction != None and self.lastState != None:
            #Get reward
            reward = self.computeReward(self.lastState,state) + self.updateReward(state)
            #Update Q value of last state-action pair
            self.learn(self.lastState, self.lastAction, reward, state)
        # Save current state as history state for next step
        self.setLastState(state)

        # Exploration: return the random choice
        if util.flipCoin(self.epsilon):
            actionChosen = random.choice(legal)
        # Exploitation: when tarining, apply count based exploration,
        # when not training, return best move
        else:
            if self.epsilon == 0:
                actionChosen = self.getBestMove(state)
            else:
                actionChosen = self.getExplorationMove(state)
        # Save current action as history action for next step
        self.setLastAction(actionChosen)
        return actionChosen

    def getBestMove(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Set the initial bestmove is None
        BestMove = None
        # Return the max Qvalue based on state
        MaxQvalue = self.maxQValue(state)
        for action in legal:
            Qvalue = self.getQValue(state,action)
            # Find which action's Qvalue is the max, and return this action
            if Qvalue == MaxQvalue:
                BestMove = action
            else:
                continue
        return  BestMove

    def getExplorationMove(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Set the initial exploration move is None
        ExplorationMove = None
        # Set the initial exploration value is negative infinity
        max_value = float('-inf')
        for action in legal:
            Qvalue = self.getQValue(state,action)
            counts = self.getCount(state,action)
            exploration_value = self.explorationFn(Qvalue,counts)
            # Find which action's exploration value is the max, and return this action
            if exploration_value > max_value:
                max_value = exploration_value
                ExplorationMove = action
            else:
                continue
        # Update the counts of the chosen move
        self.updateCount(state,ExplorationMove)
        return ExplorationMove


#################################################################################################3

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        self.lastState = None
        self.lastAction = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

'''
python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
python pacman.py -p RandomAgent -n 10 -l smallGrid
'''