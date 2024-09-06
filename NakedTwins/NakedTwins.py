import csv

def assign_value(values, box, value):
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        ASSIGNMENTS.append(values.copy())
    return values


def naked_twins(values):
    # Find all instances of naked twins in every unit
    for unit in UNIT_LIST:
        all_unit_values = [values[box] for box in unit]

        def nt_filter(value, auv=all_unit_values): return (True if (len(value) == 2 and
                                                                    auv.count(value) == 2)
                                                           else False)
        all_naked_twins = set(filter(nt_filter, all_unit_values))
        # Eliminate the naked twins as possibilities for their peers
        digits_to_remove = "".join(
            naked_twin for naked_twin in all_naked_twins)
        for box in unit:
            if values[box] not in all_naked_twins:
                for digit in digits_to_remove:
                    values = assign_value(
                        values, box, values[box].replace(digit, ""))
    return values

def cross(row_names, col_names):
    return [row + col for row in row_names for col in col_names]

def grid_values(grid):
    return {box: possibility if possibility != "0" else "123456789"
            for box, possibility in zip(cross("ABCDEFGHI", "123456789"), grid)}

def display(values):
    if not values:
        print("Not solvable")
        return
    max_widths = {col: len(max([values[row + col] for row in "ABCDEFGHI"], key=len))
                  for col in "123456789"}
    row_dic = {0: "A", 1: "B", 2: "C", 4: "D",
               5: "E", 6: "F", 8: "G", 9: "H", 10: "I"}
    output_lines = []
    for i in range(11):
        if i not in row_dic:
            line = ""
            for cols in ["123", "456", "789"]:
                for col in cols:
                    line += "" * (max_widths[col] + 1)
                line += ""
        else:
            line = ""
            row = row_dic[i]
            for cols in ["123", "456", "789"]:
                for col in cols:
                    max_width = max_widths[col]
                    padding = max_width - len(values[row+col])
                    front_padding = "" * int(padding/2)
                    back_padding = "" * (padding - int(padding/2))
                    line += front_padding + \
                        values[row + col] + back_padding + ""
                line += ""
        output_lines.append(line)
    return "".join(output_lines)

def eliminate(values):
    single_values = [box for box in BOXES if len(values[box]) == 1]
    for box in single_values:
        for peer in PEERS[box]:
            values = assign_value(
                values, peer, values[peer].replace(values[box], ""))
    return values

def only_choice(values):
    for unit in UNIT_LIST:
        for digit in "123456789":
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values = assign_value(values, dplaces[0], digit)
    return values

def reduce_puzzle(values):
    able_to_reduce = True
    while able_to_reduce:
        previous_values = values.copy()
        values = eliminate(values)  # eliminate obvious clashes
        values = only_choice(values)  # eliminate only choices
        # eliminate naked twin values from all other boxes
        values = naked_twins(values)
        # if values have changed, then there's a chance that we can further reduce puzzle
        able_to_reduce = values != previous_values
        # Sudoku is not solvable if there are any empty boxes
        if any(len(values[box]) == 0 for box in BOXES):
            return False
    return values

def search(values):
    values = reduce_puzzle(values)

    # values can now be a dict or a bool depending on whether it's solvable further
    if not values:
        return False

    if all(len(value) == 1 for value in values.values()):
        return values

    # best choice to expand the tree is the one containing least possibilities
    best_key, best_choices = min(
        values.items(), key=lambda t: len(t[1]) if len(t[1]) > 1 else 10)
    # iterating all the possible values and searching recursively
    for choice in best_choices:
        values_copy = values.copy()
        values_copy[best_key] = choice
        values_copy = search(values_copy)
        if values_copy and all(len(values_copy[box]) == 1 for box in values_copy):
            return values_copy

def solve(grid):
    values = grid_values(grid)  # convert grid into a dictionary form
    # using depth first search search for solutions recursively
    values = search(values)
    return values

ASSIGNMENTS = []

ROWS = "ABCDEFGHI"
COLS = "123456789"
BOXES = cross(ROWS, COLS)
ROW_UNITS = [cross(row, COLS) for row in ROWS]
COL_UNITS = [cross(ROWS, col) for col in COLS]
BOX_UNITS = [cross(rows, cols)
             for rows in ["ABC", "DEF", "GHI"]
             for cols in ["123", "456", "789"]]
DIAG_UNIT = [row+col for row, col in zip("ABCDEFGHI", "123456789")]
ANTI_DIAG_UNIT = [row+col for row, col in zip("ABCDEFGHI", "987654321")]
UNIT_LIST = ROW_UNITS + COL_UNITS + BOX_UNITS
UNITS = {box: [unit for unit in UNIT_LIST if box in unit] for box in BOXES}
PEERS = {box: set(box2 for unit in UNITS[box] for box2 in unit) - set([box])
         for box in BOXES}

if __name__ == '__main__':
    import time
    initial_time = time.time()
    result = 0
    i = 0
    with open('../sudoku.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            if i == 10000 :
                break
            diag_sudoku_grid = row[0]
            solution = row[1]
            sudoku_solved = display(solve(diag_sudoku_grid))
            if solution == sudoku_solved :
                result += 1
            i += 1
    result = int(result/i*100)
    final_time = round(time.time()- initial_time,2)
    print(f"It took {final_time} seconds to solve {i: ,} puzzles.")
    print(f"Accuracy : {result}%")