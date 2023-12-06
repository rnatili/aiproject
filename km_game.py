#!/usr/bin/env python3

from collections import namedtuple, Counter, defaultdict
import random
import math
import functools
import itertools
import time
from math import sqrt, log
import csv
import numpy as np

#Code from Assigment 3 utilized for efficiency from Professor Flynn
#As well as it's original state from AIMA
#Parts used for efficient and easily testable game engine:
    # GameState
    # Game
    # Isola (modified to this game)
    # play_game
    # Searches

GameState = namedtuple('GameState', 'board, to_move')

# PJF - removed the play_game method in the original AIMA version of this class
# to avoid confusion with separate play_game function further down.

# This is an abstract class

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)


#Trey Natili, modified from PJF
class KnightMoves(Game):
    """
    A concrete subclass of Game, implementing the Knight Moves game.

    We're representing the board as a 1D character string, with legal characters
    [' ', '1', '2', 'B'] representing [open, player-1, player-2, blocked].

    An example 4x3 board
    BB1
    2 B
     BB
    BBB

    would be represented in the boardstring 'BB12 B BBBBB'

    So a 4x3 game has 1D cell indices of
    +--+--+--+
    | 0| 1| 2|     [0][0] cell has 1D index 0...
    +--+--+--+
    | 3| 4| 5|
    +--+--+--+
    | 6| 7| 8|
    +--+--+--+
    | 9|10|11|     [3][2] cell has index 11.
    +--+--+--+

    A 1D cell index is converted back to [row][column] 2D board coordinates as:
     row = floor(index/ncolumns)
     col = index MOD ncolumns
    Python has a builtin function divmod() to do this.

    The 1D cell indices have some efficiency and other practical advantages over 2D for
    this example.
    """
    def __init__(self, nr:int=4, nc:int=4):
        assert (nr>0) and (nc>0) and (nr<20) and (nc<20), \
               "Error: silly board size."
        self.nr = nr
        self.nc = nc
        #
        # Define the possible moves for a player. These are expressed as
        # offsets to the player's current position
        # this default set considers moves up, down, left, and right.
        self.moves = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]

        board = ' ' * (nr*nc)
        self.start = 0
        self.initial = GameState(board=board, to_move='1')

    # utilities to convert 2D board coordinates to 1D boardstring indices and back.
    # could memoize these using the @functools.cache decorator
    def _ix2D(self,ix):
        return divmod(ix,self.nc)

    def _ix1D(self,i,j):
        return i*self.nc+j

    # could also memoize this
    def legalCell(self,row,col):
        """ returns True if the (row,column) position is legal for this board,
        regardless of whether the cell is occupied.
        """
        return (row>=0) and (col>=0) and (row<self.nr) and (col<self.nc)

    # useful generic functions
    def occupied(self,board,row,col):
        """Returns True if the specified cell is occupied (by a player) or blocked and False if not."""
        assert self.legalCell(row,col),'occupied: Illegal board cell location ({0},{1})'.format(row,col)
        return board[self._ix1D(row,col)] != ' '

    def prettyboard(self,board):
        """Pretty-printable string representation of an Isola game instance."""
        s = ''
        # header
        strs = ['{0: >2}'.format(i) for i in range(max(self.nr,self.nc))]
        for i in range(2):
            s += '   ' + ''.join([x[i] for x in strs[:self.nc]]) + '\n'
        s += '  +' + '-'*self.nc + '+\n'
        for i in range(self.nr):
            s += '{0}|'.format(strs[i]) + board[self._ix1D(i,0):self._ix1D(i+1,0)] + '|\n'
        s += '  +' + '-'*self.nc + '+\n'
        return s

    def display(self, state):
        """Print or otherwise display the state."""
        print('State: {0}'.format(state))
        print('Board:')
        print(self.prettyboard(state.board))

    def _findPlayer(self,board,player):
        """
        Find the 2D position of the player (or [-1,-1] if the player isn't on the board yet).
        This is ICKY. Would be better to put this
        in the game state somewhere but the parent class doesn't allow it.
        """
        i1d = board.find(player)
        if i1d != -1:
            return self._ix2D(i1d)
        else:
            return [-1,-1]

    def _legalMoves(self,board,player):
        """
        Return the legal moves (as a list of positions (row',col'))
        for the player ('1' or '2')
        """
        lm = []
        # find the player on the board. Handle initial turn when they haven't
        # been placed.
        r,c = self._findPlayer(board,player)
        if (r<0) or (c<0): # first turn
            ret = list(itertools.product(range(self.nr), range(self.nc)))
            if player == '1': # can move anywhere
                return ret
            elif player == '2': # move anywhere except where 1 is
                r1,c1 = self._findPlayer(board,'1')
                ret.remove((r1,c1))  # can't move there already occupied
                return ret
        for m in self.moves:
            # compute new position (add offset specified in the move)
            # then see if it's on the board and not occupied.
            rn,cn = [r+m[0],c+m[1]]
            if self.legalCell(rn,cn) and not self.occupied(board,rn,cn):
                # add it to the set of moves returned.
                lm.append((rn,cn))
        return lm

    def actions(self,state):
        """available actions for the current state of game
        Return the legal moves (as a list of positions (row',col'))
        for the player ('1' or '2')
        """
        lm = []
        # find the player on the board. Handle initial turn when they haven't
        # been placed.
        player = state.to_move
        r,c = self._findPlayer(state.board,player)
        #print('a: rc {0} state {1}'.format([r,c],state))
        if (r<0) or (c<0): # first turn
            #print("a: first turn")
            ret = list(itertools.product(range(self.nr), range(self.nc)))
            if player == '1': # can move anywhere
                #print('a: p1 initial: returning {0}'.format(ret))
                return ret
            elif player == '2': # move anywhere except where 1 is
                r1,c1 = self._findPlayer(state.board,'1')
                #print('a: 2: r1c1 {0}'.format([r1,c1]))
                ret.remove((r1,c1))  # can't move there already occupied
                #print('a: 2: returning {0}'.format(ret))
                return ret
        for m in self.moves:
            # compute new position (add offset specified in the move)
            # then see if it's on the board and not occupied.
            rn,cn = [r+m[0],c+m[1]]
            if self.legalCell(rn,cn) and not self.occupied(state.board,rn,cn):
                # add it to the set of moves returned.
                lm.append((rn,cn))
        #print('a: normal: returning {0}'.format(lm))
        return lm


    def result(self, state, move):
        """Return the state that results from making a move from a different state."""
        p = state.to_move # the player
        b = state.board   # the board
        [rnew,cnew] = move # the new position
        inew = self._ix1D(rnew,cnew)
        # change current player location to a 'B' (blocked) if after 1st turn
        rold,cold = self._findPlayer(b,p)
        iold = self._ix1D(rold,cold) # negative means first turn
        bl = list(state.board)
        if iold >= 0: bl[iold]='B'  # only after first turn
        bl[inew]=p
        b = ''.join(bl)
        next_player = '1' if p == '2' else '2'
        gs = GameState(board=b,to_move=next_player)
        #print("=Action consideration=")
        #print("in state {0}".format(state))
        #print("proposed action {0}".format(move))
        #print("result {0}".format(gs))
        #_ = input("Enter x: ")
        return gs

    def utility(self, state, player):
        """
        Return the value of this final state to player.
        If this is called, we know that one of the players in the state has no moves.
        Since any heuristic we develop should try to estimate this at a nonterminal node, let's clip
        the values between (max number of possible moves) and -(max number of possible moves)
        """
        assert player in '12', "Error: illegal player {0}".format(player)
        #print('U:Req utility of {0} for player {1}'.format(state,player))
        #print(self.prettyboard(state.board))
        nm = []
        for p in '12':
            stmp = GameState(board=state.board,to_move=p)
            nm.append(len(self.actions(stmp)))
        #print('U: moves by player: {0}'.format(nm))
        assert nm[0]*nm[1] == 0,"U: Error - One of the counts should be 0."
        lsm = len(self.moves)
        if ((player=='1') and (nm[0]>nm[1])) or ((player=='2') and (nm[1]>nm[0])):
            u = lsm
        elif nm[0]==nm[1]:
            if (player == '1'):
                u = -lsm
            else:
                u = lsm
        else:
            u = -lsm
        #print(f"U: returning u={u}")
        return u

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        #print('tt: state is {0}'.format(state))
        return len(self.actions(state)) == 0

    def is_terminal(self,state): return self.terminal_test(state)


# PJF added more verbosity for greater debuggingosity.
#
def play_game(game, strategies: dict, verbose=False):
    """Play a turn-taking game. `strategies` is a {player_name: function} dict,
    where function(state, game) is used to get the player's move."""
    state = game.initial
    if verbose:
        print("initial state")
        game.display(state)
    while not game.is_terminal(state):
        player = state.to_move
        move = strategies[player](game, state)
        state = game.result(state, move)
        game.start = game.start + 1
        if verbose:
            print('Player', player, 'move:', move)
            print(game.prettyboard(state.board))
    uf = game.utility(state,'1')
    #uf = game.utility(state, player)
    if verbose:
        print('End-of-game state')
        game.display(state)
        #print('End-of-game utility: {0}'.format(uf))
    return state,uf


def minimax_search(game, state):
    """Search game tree to determine best move; return (value, move) pair."""

    player = state.to_move

    def max_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a))
            if v2 > v:
                v, move = v2, a
        return v, move

    def min_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a))
            if v2 < v:
                v, move = v2, a
        return v, move

    return max_value(state)

infinity = math.inf

def alphabeta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    def max_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    def min_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move
    #print (max_value(state, -infinity, +infinity))
    return max_value(state, -infinity, +infinity)

###UPDATE 3: ATTEMPT AT A MONTE CARLO TREE SEARCH### UPDATE 4: SUCCESS AT A MCTS
#Using what I learned online, best approach is to create a node class, and also to use a method
#called Upper Confidence Bound for Trees to determine when to stop exploring and to make a decision (exploitation)
#As a result, I'm going to implement that

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.utility = 0


def monte_carlo_tree_search(game, state, iterations = 50, eweight = 1.4):
    root = Node(state)

    def select(node):
        if not node.children:
            return node

        log_total = log(sum(child.visits for child in node.children))
        #score = lambda child: child.utility / child.visits + 1.4 * sqrt(log_total / child.visits)
        score = lambda child: (child.utility / max(1,child.visits) + eweight * sqrt(log_total / max(1, child.visits)))
        return max(node.children, key=score)

    def expand(node):
        actions = game.actions(node.state)
        for action in actions:
            new_state = game.result(node.state, action)
            new_node = Node(new_state, parent=node)
            node.children.append(new_node)
        return node.children[0] if node.children else node

    def simulate(node): #Currently just executes random simulation
        simulation_state = node.state
        while not game.is_terminal(simulation_state):
            action = random.choice(game.actions(simulation_state))
            simulation_state = game.result(simulation_state, action)
        return game.utility(simulation_state, node.state.to_move)

    def backpropagate(node, result):
        while node is not None:
            node.visits += 1
            node.utility += result
            node = node.parent

    for _ in range(iterations): #The core functionality, runs X times
        selected_node = select(root)
        expanded_node = expand(selected_node)
        simulation_result = simulate(expanded_node)
        backpropagate(expanded_node, simulation_result)

    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.utility, game.actions(state)[root.children.index(best_child)]


def random_player(game, state): return random.choice(list(game.actions(state)))

def player(search_algorithm, limit = None, weight = None):
    """A game player who uses the specified search algorithm"""
    #Kinda a gross way to incorporate paramaterized players
    if limit is None and weight is None:
        return lambda game, state: search_algorithm(game, state)[1]
    elif weight is None:
        return lambda game, state: search_algorithm(game, state, iterations = limit)[1]
    elif limit is None:
        return lambda game, state: search_algorithm(game, state, eweight = weight)[1]
    else:
        return lambda game, state: search_algorithm(game, state, limit, weight)[1]

def human_player(game, state): #Text input based player
    num = 0
    actions = list(game.actions(state))
    if (game.start < 2):
        startrow = input("Choose a starting row: ")
        startcol = input("Choose a starting column: ")
        return (int(startrow),int(startcol))
    else:
        for action in actions:
            num = num + 1
            print(num, ':', action)
        

    move = input("Choose the number of the move you'd like to make: ")
    return list(game.actions(state))[int(move) - 1]

def main():
    #'''#There are three sets of these comment outs, removing the # before them allows easy switching between code functionality
    #Player control option
    print("Player options: \n 1. Random Moves \n 2. Mini-Max Search \n 3. Alpha-Beta Search \n 4. Easy Opponent \n 5. Medium Opponent \n 6. Hard Opponent \n 7. Human Player")
        
    player1input = input("Choose the number for Player 1: ")
    player2input = input("Choose the number for Player 2: ")
    
    player_functions = {
        '1': random_player,
        '2': player(minimax_search),
        '3': player(alphabeta_search),
        '4': player(monte_carlo_tree_search, 6, 1.65),
        '5': player(monte_carlo_tree_search, 12, 1.65),
        '6': player(monte_carlo_tree_search, 25, 1.65),
        '7': human_player,
    }
    
    #Board size is hard coded as it likely would not need to be changed often, and requires two extra inputs per run.
    #LOOK HERE: Change board size below by changing nr or nc
    result = play_game(KnightMoves(nr = 6, nc = 6), \
                    {'1': player_functions[player1input], '2': player_functions[player2input]}, verbose = True)
    if result[1] == 0:
        print('Draw')#Should never happen

    elif result[1] < 0:
        print('Player 2 Wins')
    else:
        print('Player 1 Wins')


'''#Second comment set, this one never needs commented out
#Rapid Test Option
#This mode is highly experimental and requires code tweaking to be used properly. 
#As it is set up when I turn it in, it'll generate a CSV file for the most important graph of the report.

    csv_filename = 'results.csv'
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Limit', 'Win Percentage'])

    #For noninteger step sizes, such as testing weights. Used for the first for loop.
    start = 1
    end = 2
    step = 0.025
    values = np.arange(start, end + step, step)

    
    for limit in range(1,200,5): #This iterates through different paramater settings, can be changed to iterate others
        p1w = 0
        p2w = 0
        print("Limit testing " + str(limit))
        for i in range(100): #This runs trials per setting
            #print("Running simulation " + str(i))
            result = play_game(KnightMoves(nr = 8, nc = 8), \
                        {'1': random_player, '2': player(monte_carlo_tree_search,limit)}, verbose = False)
            if result[1] == 0:
                print("A tie happened somehow")
            elif result [1] < 0:
                p2w = p2w + 1
                #print('Player 2 wins')
            else:
                p1w = p1w + 1
                #print('Player 1 wins')

        #print('Player 1 wins: ' + str(p1w))
        #print('Player 2 wins: ' + str(p2w))
        win_percentage = (p2w / (p1w + p2w)) * 100

        csv_writer.writerow([limit, win_percentage])
        csv_file.flush()

    

    #print(f"Player one won {p1w} times and player two won {p2w} times.")

'''#Comment this out if you want the second mode

if __name__ == "__main__":
    main()
