"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import math
from random import randint

weights = [1, 1]

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def percent_game_completed(low, high, game):
    """Helper method to check what percent of the game is completed
    """
    percent = game.move_count/(game.height * game.width)*100
    if low <= percent and high > percent:
        return True
    return False

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    y2, x2 = game.get_player_location(game.get_opponent(player))

    distance_to_center = float((h - y)**2 + (w - x)**2)
    opp_distance_to_center = float((h - y2)**2 + (w - x2)**2)
    return float((own_moves - opp_moves) + (opp_distance_to_center - distance_to_center)*0.634/(game.move_count))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    pos_y, pos_x = game.get_player_location(player)
    number_of_boxes_to_center = abs(pos_x - math.ceil(game.width/2)) + \
        abs(pos_y - math.ceil(game.height/2)) - 1

    if percent_game_completed(0, 10, game):
        return 2*own_moves - 0.5*number_of_boxes_to_center
    elif percent_game_completed(10, 40, game):
        return float(3*own_moves - opp_moves - 0.5*number_of_boxes_to_center)
    else:
        return 2*own_moves - opp_moves


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    own_moves = len(my_moves)
    opp_moves = len(opponent_moves)
    moves_so_far = 0
    for box in game._board_state:
        if box == 1:
            moves_so_far += 1    

    w, h = game.get_player_location(game.get_opponent(player))
    y, x = game.get_player_location(player)
    distance_to_center = float((h - y)**2 + (w - x)**2)

    wall_boxes = [(x, y) for x in (0, game.width-1) for y in range(game.width)] + \
    [(x, y) for y in (0, game.height-1) for x in range(game.height)]

    penalty = 0
    if (x, y) in wall_boxes:
        penalty -= 1

    quality_of_move = 0
    for move in my_moves:
        y, x = move
        dist = float((h - y)**2 + (w - x)**2)
        if dist == 0:
            quality_of_move += 1
        else:
            quality_of_move += 1/dist
        if move in opponent_moves:
            quality_of_move -= 1

    if percent_game_completed(0, 40, game):
        return float(own_moves - opp_moves - distance_to_center + quality_of_move + penalty)
    else:       
        return own_moves - 2*opp_moves


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=50.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def __min_value(self, game, depth):
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        min_val = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_val = min(min_val, self.__max_value(forecast, depth - 1))
        return min_val

    def __max_value(self, game, depth):
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        max_val = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_val = max(max_val, self.__min_value(forecast, depth - 1))
        return max_val

    def __is_terminal(self, game, depth):
        """Helper method to check if we've reached the end of the game tree or
        if the maximum depth has been reached.
        """
        self.__check_time()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def minimax(self, game, depth):
        """Implementation of depth-limited minimax search algorithm.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        self.__check_time()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        vals = [(self.__min_value(game.forecast_move(m), depth - 1), m) for m in legal_moves]
        _, move = max(vals)
        return move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves(self)
        if len(legal_moves) > 0:
            best_move = legal_moves[randint(0, len(legal_moves)-1)]
        else:
            best_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while True:
                current_move = self.alphabeta(game, depth)
                if current_move == (-1, -1):
                    return best_move
                else:
                    best_move = current_move
                depth += 1
        except SearchTimeout:
            return best_move
        return best_move

    def __max_value(self, game, depth, alpha, beta):
        self.__check_time()
        best_move = (-1, -1)
        if self.__is_terminal(game, depth):
            return (self.score(game, self), best_move)
        value = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            result = self.__min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if result[0] > value:
                value, _ = result
                best_move = move
            if value >= beta:
                return (value, best_move)
            alpha = max(alpha, value)
        return (value, best_move)

    def __min_value(self, game, depth, alpha, beta):
        self.__check_time()
        best_move = (-1, -1)
        if self.__is_terminal(game, depth):
            return (self.score(game, self), best_move)
        value = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            result = self.__max_value(game.forecast_move(move), depth - 1, alpha, beta)
            if result[0] < value:
                value, _ = result
                best_move = move
            if value <= alpha:
                return (value, best_move)
            beta = min(beta, value)
        return (value, best_move)

    def __is_terminal(self, game, depth):
        self.__check_time()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """  
        self.__check_time()
        _, move = self.__max_value(game, depth, alpha, beta)
        return move