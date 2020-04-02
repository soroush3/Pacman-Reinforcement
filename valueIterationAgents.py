# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util
import copy

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
    
    def runValueIteration(self):
        # Write value iteration code here
        # OFFLINE PLANNER
        states = self.mdp.getStates()
        # iterations to update state values
        for i in range(self.iterations):
            vals = copy.deepcopy(self.values)
            for state in states:
                # terminal states are always of value 0
                if self.mdp.isTerminal(state):
                    continue
                qValList = []
                for action in self.mdp.getPossibleActions(state):
                    qVal = self.computeQValueFromValues(state, action)
                    qValList.append(qVal)
                vals[state] = max(qValList)
            self.values = vals

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getDirection(self, startState, endState):
        """
        Returns the direction taken to get to endState from startState
        """
        # states are the same, no direction to move
        if startState == endState:
            return None
        # move along y axis, north or south
        if startState[0] == endState[0]:
            if endState[1] > startState[1]:
                return "north"
            return "south"
        else:
            if endState[0] > startState[0]:
                return "east"
            return "west"

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return 0
        qSum = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            # probability is 0, continue
            if transition[1] == 0:
                continue
            reward = self.mdp.getReward(state, action, transition[0])
            qSum += (transition[1] * (reward + self.discount \
                                            * (self.getValue(transition[0]))))
        return qSum

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        actionList = []
        for act in self.mdp.getPossibleActions(state):
            qVal = self.computeQValueFromValues(state, act)
            actionList.append((act, qVal))
        best = max(actionList, key=lambda x:x[1])
        return best[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Update only one state per iteration
        """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):

            state = states[i % len(states)]
            vals = copy.deepcopy(self.values)

            if self.mdp.isTerminal(state):
                continue

            qValList = []
            for action in self.mdp.getPossibleActions(state):
                qVal = self.computeQValueFromValues(state, action)
                qValList.append(qVal)
            vals[state] = max(qValList)
            self.values = vals

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # compute predecessors of all states
        preds = {}
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            possActions = self.mdp.getPossibleActions(state)

            for action in possActions:
                for trans, prob in self.mdp.getTransitionStatesAndProbs(state,
                                                                        action):
                    # looking for nonzero probabilities
                    if prob == 0:
                        continue
                    elif trans not in preds:
                        preds[trans] = set()
                    preds[trans].add(state)

        q = util.PriorityQueue()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state): 
                continue
            qValList = []
            for action in self.mdp.getPossibleActions(state):
                qVal = self.computeQValueFromValues(state, action)
                qValList.append(qVal)

            maxQVal = max(qValList)
            
            diff = abs(self.values[state] - maxQVal)
            q.update(state, -diff)

        for i in range(self.iterations):
            if q.isEmpty():
                break
            
            state = q.pop()
            # update self.values[state]
            if not self.mdp.isTerminal(state):
                qListVal = []
                for action in self.mdp.getPossibleActions(state):
                    qVal = self.computeQValueFromValues(state, action)
                    qListVal.append(qVal)
                self.values[state] = max(qListVal)
            
            for predecessor in preds[state]:
                if self.mdp.isTerminal(predecessor):
                    continue
                pVal = self.values[predecessor]
                qListVal = []
                for action in self.mdp.getPossibleActions(predecessor):
                    qVal = self.computeQValueFromValues(predecessor, action)
                    qListVal.append(qVal)

                diff = abs(pVal - max(qListVal))
                if diff > self.theta:
                    q.update(predecessor, -diff)
