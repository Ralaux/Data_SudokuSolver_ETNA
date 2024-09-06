import pandas as pd
import time

N = 9

def printing(arr, truth):
    stri = ""
    for i in range(N):
        for j in range(N):
            stri += str(arr[i][j])
    if stri == truth:
        return 1
    return 0

def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
    for x in range(9):
        if grid[x][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def solveSudoku(grid, row, col):
    if (row == N - 1 and col == N):
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False


def TestAll(size_test):
    final = 0
    start = time.time()
    for x in range(0, size_test) :
        input_line = data.values[x][0]
        grid = []
        for i in range (0,9):
            grid.append([])
        for j in range(0,81):
            grid[int(j/9)].append(int(input_line[j]))
        
        if (solveSudoku(grid, 0, 0)):
            final += printing(grid, data.values[x][1])
        else:
            print("no solution  exists ")    
    end = time.time()
    if sudoku_nb < 1000 :
        print(f"Sample of {sudoku_nb} SUDOKUS")
    else :
        print(f"Sample of {sudoku_nb: ,} SUDOKUS")
    print("Score = ", final * 100 / sudoku_nb, "%")
    print("Duration = ", round(end - start, 2), "s")
    
    
data = pd.read_csv("../sudoku.csv")

nb_range = [100, 1000, 10000]
for sudoku_nb in nb_range :
    TestAll(sudoku_nb)