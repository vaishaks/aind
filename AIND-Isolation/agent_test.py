"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import sample_players

from importlib import reload

class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer(2, sample_players.improved_score)
        self.player2 = sample_players.GreedyPlayer()
        self.game = isolation.Board(self.player1, self.player2, 9, 9)
        self.time_left = lambda : float("inf")

    def test_minimax_next_move(self):
        self.game._board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 50]
        print(self.game.to_string())
        curr_move = self.game._active_player.get_move(self.game.copy(), self.time_left)
        self.assertEqual((3, 6), curr_move)

if __name__ == '__main__':
    unittest.main()
