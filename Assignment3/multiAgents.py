# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isLose():
            return -float('inf')

        for i, ghost_state in enumerate(newGhostStates):
            ghost_pos = ghost_state.getPosition()
            if ghost_state.scaredTimer == 0:  # Ghost is dangerous
                dist_to_ghost = abs(newPos[0] - ghost_pos[0]) + abs(newPos[1] - ghost_pos[1])
                if dist_to_ghost < 2:
                    return -float('inf')  # Immediate danger!

        # 3. FOOD INCENTIVE
        # Find the distance to the closest food pellet
        food_list = newFood.asList()
        min_food_dist = float('inf')

        if len(food_list) > 0:
            for food in food_list:
                # Calculate Manhattan distance to each food pellet
                dist = abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
                if dist < min_food_dist:
                    min_food_dist = dist
        else:
            min_food_dist = 0  # No food left, we effectively won this step

        # Calculate a score based on the reciprocal of the distance.
        # We add 1 to the divisor to avoid division by zero.
        # Closer food = Higher value (e.g., dist 1 -> 1.0, dist 2 -> 0.5)
        food_score = 10.0 / (min_food_dist + 1)

        # 4. GAME SCORE
        # We still want the base game score because it accounts for food already eaten
        current_score = successorGameState.getScore()

        # Combine the factors
        return current_score + food_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getValue(self, state, index, depth):
        # Terminal states or depth reached
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        # Pacman's turn (Maximizer)
        if index == 0:
            return self.maxValue(state, index, depth)
        # Ghosts' turn (Minimizers)
        else:
            return self.minValue(state, index, depth)

    def maxValue(self, state, index, depth):
        v = -float('inf')
        actions = state.getLegalActions(index)
        for action in actions:
            successor = state.generateSuccessor(index, action)
            # After Pacman (0), move to Ghost 1 at the same depth
            v = max(v, self.getValue(successor, 1, depth))
        return v

    def minValue(self, state, index, depth):
        v = float('inf')
        actions = state.getLegalActions(index)
        numAgents = state.getNumAgents()
        nextAgent = index + 1
        nextDepth = depth

        # If this was the last ghost, the next agent is Pacman at a deeper level
        if nextAgent == numAgents:
            nextAgent = 0
            nextDepth += 1

        for action in actions:
            successor = state.generateSuccessor(index, action)
            v = min(v, self.getValue(successor, nextAgent, nextDepth))
        return v
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        bestScore = -float('inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = self.getValue(gameState.generateSuccessor(0, action), 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getValue(self, state, index, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        if index == 0:
            return self.maxValue(state, index, depth, alpha, beta)
        else:
            return self.minValue(state, index, depth, alpha, beta)
    def maxValue(self, state, index, depth, alpha, beta):
        v = -float('inf')
        for action in state.getLegalActions(index):
            v = max(v, self.getValue(state.generateSuccessor(index, action), 1, depth, alpha, beta))
            if v > beta: return v  # Prune
            alpha = max(alpha, v)
        return v

    def minValue(self, state, index, depth, alpha, beta):
        v = float('inf')
        numAgents = state.getNumAgents()
        nextAgent = index + 1
        nextDepth = depth
        if nextAgent == numAgents:
            nextAgent = 0
            nextDepth += 1

        for action in state.getLegalActions(index):
            v = min(v, self.getValue(state.generateSuccessor(index, action), nextAgent, nextDepth, alpha, beta))
            if v < alpha: return v  # Prune
            beta = min(beta, v)
        return v
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -float('inf')
        beta = float('inf')
        bestScore = -float('inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            score = self.getValue(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            # Update alpha at the root too!
            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction




# Questions
"""
36
Because there are multiple ghosts, every ghost must take a turn before Pacman does. Thus there are multiple minimizers.
Layer 0 (Max): Pac man chooses a move
Layer 1 (Min): Ghost 1 chooses a move
Layer 2 (Min)L Ghost 2 chooses a move

Even though this is depth of 1. the tree is 2 layers deep as it has to consider 3 agent's worth of moves.


37.
Alphabeta solves the problem of exponential increase in states, it prunes of moves that are obviously bad by comparing to other parts of the tree. This eliminates unnecessary work.

38.
alpha and beta can be though of as current best guarantees for each side.
alpha would be for Pacman's current best (highest, floor) achievable score. Thus anything lower can be safely pruned
beta would be for the ghosts current best (lowest meaning ceiling) achievable score. Thus anything high can be safely pruned.

If alpha >= beta it means that the ghost has found a way to make a brnach so bad for Pacman that Pacman would never choose it since there is a better state somewhere else in the tree
(its where alpha was updated or assigned)
"""