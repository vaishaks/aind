from utils import *

def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    values = reduce_puzzle(values)
    if values == False:
        return False
    solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
    if solved_values_after == 81:
        return values
    l, least_possibility = min((len(values[id]), id) for id in values if len(values[id]) > 1)
    for val in values[least_possibility]:
        values_temp = values.copy()
        values_temp[least_possibility] = val
        solution = search(values_temp)
        if solution:
            return solution

values_easy = grid_values('..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..')
values_hard = grid_values('4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......')
display(search(values_hard))