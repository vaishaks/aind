# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  from collections import defaultdict
  actions = util.Stack()
  frontier = util.Stack()
  explored = []  
  parent_map = defaultdict(lambda: None)
  frontier.push((problem.getStartState(), "Stop", 1))
  while True:
    if frontier.isEmpty():
      return actions
    (leaf_node, direction, val) = frontier.pop()
    if problem.isGoalState(leaf_node):
      current = (leaf_node, direction, val)
      while current != None:
        current_node, current_direction, curret_val = current
        actions.push(current_direction)
        current = parent_map[current_node]
      return actions.list[::-1]
    explored.append(leaf_node)
    children = problem.getSuccessors(leaf_node)
    for (child_node, child_direction, child_val) in children:
      frontier_list = [frontier_node for (frontier_node, frontier_node_direction, frontier_node_val) in frontier.list]
      if child_node not in frontier_list and child_node not in explored:
        frontier.push((child_node, child_direction, child_val))
        parent_map[child_node] = (leaf_node, direction, val)

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
  "*** YOUR CODE HERE ***"
  from collections import defaultdict
  from game import Directions
  node = problem.getStartState()
  if problem.isGoalState(node):
    return [Directions.STOP]
  frontier = util.Queue()
  frontier.push((node, [Directions.STOP]))
  explored = util.Stack()
  actions = util.Stack()
  parent_map = defaultdict(lambda: None)
  while True:
    if frontier.isEmpty():
      return []
    node, node_actions = frontier.pop()
    explored.push(node)    
    children = problem.getSuccessors(node)
    for (child_node, child_direction, child_cost) in children:
      frontier_list = [frontier_node for (frontier_node, frontier_actions) in frontier.list]
      if child_node not in explored.list and child_node not in frontier_list:
        if problem.isGoalState(child_node):
          return node_actions + [child_direction]
        frontier.push((child_node, node_actions + [child_direction]))
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  from collections import defaultdict
  from game import Directions
  node = problem.getStartState()
  frontier = util.PriorityQueue()
  frontier.push((node, Directions.STOP, 0), 0)
  explored = util.Stack()
  actions = util.Stack()
  parent_map = defaultdict(lambda: None)
  while True:
    if frontier.isEmpty():
      return []
    node, node_direction, node_cost = frontier.pop()
    if problem.isGoalState(node):
      current = (node, node_direction, node_cost)
      while current != None:
        current_node, current_direction, curret_val = current
        actions.push(current_direction)
        current = parent_map[current_node]                      
      actions.list.remove(Directions.STOP)
      return actions.list[::-1]
    explored.push(node)
    children = problem.getSuccessors(node)
    for (child_node, child_direction, child_cost) in children:
      frontier_list = []
      frontier_cost = {}
      frontier_item = ""
      for (priority, (frontier_node, frontier_node_direction, frontier_node_cost)) in frontier.heap:
        frontier_list.append(frontier_node)
        frontier_cost[frontier_node] = frontier_cost
        if frontier_node == child_node:
          frontier_item = (priority, (frontier_node, frontier_node_direction, frontier_node_cost))            
      if child_node not in explored.list and child_node not in frontier_list:
        frontier.push((child_node, child_direction, child_cost+node_cost), child_cost+node_cost)
        parent_map[child_node] = (node, node_direction, node_cost)
      elif child_node in frontier_list and child_cost+node_cost < frontier_cost[child_node]:
        frontier.heap.remove(frontier_item)
        frontier.push((child_node, child_direction, child_cost+node_cost), child_cost+node_cost)      


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  from collections import defaultdict
  from game import Directions
  node = problem.getStartState()
  frontier = util.PriorityQueue()
  frontier.push((node, Directions.STOP, 0, [Directions.STOP]), 0)
  explored = util.Stack()
  actions = util.Stack()
  parent_map = defaultdict(lambda: None)
  while True:
    if frontier.isEmpty():
      return []
    node, node_direction, node_cost, node_actions = frontier.pop()
    if problem.isGoalState(node):
      print node_actions
      return node_actions
    explored.push(node)
    children = problem.getSuccessors(node)
    for (child_node, child_direction, child_cost) in children:
      frontier_list = []
      frontier_cost = {}
      frontier_item = ""
      for (priority, (frontier_node, frontier_node_direction, frontier_node_cost, frontier_node_actions)) in frontier.heap:
        frontier_list.append(frontier_node)
        frontier_cost[frontier_node[0]] = frontier_cost
        if frontier_node[0] == child_node[0]:
          frontier_item = (priority, (frontier_node, frontier_node_direction, frontier_node_cost, frontier_node_actions))            
      total_cost = (child_cost + node_cost) + heuristic(child_node, problem)
      if child_node not in explored.list and child_node not in frontier_list:        
        frontier.push((child_node, child_direction, total_cost, node_actions + [child_direction]), total_cost)
      elif child_node in frontier_list and total_cost < frontier_cost[child_node[0]]:
        frontier.heap.remove(frontier_item)
        frontier.push((child_node, child_direction, total_cost, node_actions  + [child_direction]), total_cost)  
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
