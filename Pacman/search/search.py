# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #Initialize a stack (LIFO) as a frontier with initial state as first element
    #This choice was made because DFS expands the deepest nodes first and thus the last ones we added to the stack 
    actions = [] #keeps track of pacman actions to do
    frontier = util.Stack()
    visited = [] 
    frontier.push((problem.getStartState(), []))
    
    #while frontier not empty pop the last element added (deepest) N.B. we don't care about the stepCost in this algorithm so only 
    #state and action are stored in a variable 
    while not frontier.isEmpty():
        currState, action = frontier.pop()
        #if the state popped is a goal state, stop the algorithm and return the complete list of actions reached
        if problem.isGoalState(currState):
            actions = action
            return actions
        #if not goal state:
        #this condition ensures a graph search version (avoids visiting already discovered nodes)
        if currState not in visited:
            visited.append(currState)
            #get Successors returns a list of successor, action, cost 
            for nextState in problem.getSuccessors(currState):
                newAction = action + [nextState[1]] #adding the new action to our list of actions
                nextNode = (nextState[0], newAction) #setting pacman state as successor state 
                frontier.push(nextNode) 
    return actions
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
     #Initialize a queue(FIFO) as a frontier with initial state as first element (simulating discovering branches level by level)
     #The logic behnid this is the same as DFS but the it is the data structure that changes 
    actions = []
    frontier = util.Queue() #Queue is used in order to make sure that the nodes are expanded level by level 
    #meaning we get the successor of the first nodes added first 
    visited = []
    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        currState, action = frontier.pop()
        if problem.isGoalState(currState):
            actions = action
            return actions
        if currState not in visited:
            visited.append(currState)
            for nextState in problem.getSuccessors(currState):
                newAction = action + [nextState[1]]
                nextNode = (nextState[0], newAction)
                frontier.push(nextNode)
    return actions
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
      #The logic behnid this is the same as DFS and BFS
    actions = []
    # a priority queue was used here because we need to keep track of the cost (due to the fact that this algorithm chooses the lowest cost)
    frontier = util.PriorityQueue()
    visited = []
    cost = 0;
    #pushing pacman's start state with a cost of 0
    frontier.push((problem.getStartState(), [], cost), cost)

    while not frontier.isEmpty():
        #cost is added because it is taken into consideration 
        currState, action, cost = frontier.pop()
        if problem.isGoalState(currState):
            actions = action
            return actions
        if currState not in visited:
            visited.append(currState)
            for nextState, nextAction, nextCost in problem.getSuccessors(currState):
                #adding up the cost to go to next node 
                newCost = cost + nextCost
                newAction = action + [nextAction]
                nextNode = ((nextState, newAction, newCost))
                #updating our priority queue with the new successor 
                frontier.update(nextNode, newCost)
    return actions
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #heuristic is manhattanHeuristic already implemented 
    #the logic is the same as the UCS algorithm however heuristic becomes actual_cost + heuristic 
    actions = []
    frontier = util.PriorityQueue()
    visited = []
    cost = 0;
    myHeuristic = heuristic(problem.getStartState(), problem)
    frontier.push((problem.getStartState(), [], cost), myHeuristic)

    while not frontier.isEmpty():
        currState, action, cost = frontier.pop()
        if problem.isGoalState(currState):
            actions = action
            return actions
        if currState not in visited:
            visited.append(currState)
            for nextState, nextAction, nextCost in problem.getSuccessors(currState):
                newCost = cost + nextCost 
                newAction = action + [nextAction]
                nextNode = ((nextState, newAction, newCost))
                #f(n) = g(n) + h(n) where g is the cost and h the heuristic 
                newHeuristic = newCost + heuristic(nextState, problem)
                frontier.update(nextNode, newHeuristic)
    return actions

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
