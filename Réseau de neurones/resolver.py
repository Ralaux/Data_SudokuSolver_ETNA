import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Sequential,load_model
from keras.utils import Sequence
from keras.layers import *
import matplotlib.pyplot as plt
import time

path = "../"
data = pd.read_csv(path+"sudoku.csv")

model = load_model("best_weights.hdf5")

def norm(a):
    return (a/9)-.5

def denorm(a):
    return (a+.5)*9

def inference_sudoku(sample):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    feat = sample
    
    while(1):
    
        out = model.predict(feat.reshape((1,9,9,1)),verbose = 0)  
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        feat = denorm(feat).reshape((9,9))
        mask = (feat==0)
     
        if(mask.sum()==0):
            break
            
        prob_new = prob*mask
    
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
    
    return pred

def test_accuracy(feats, labels):
    
    correct = 0
    
    for i,feat in enumerate(feats):
        
        pred = inference_sudoku(feat)
        
        true = labels[i].reshape((9,9))+1
        
        if(abs(true - pred).sum()==0):
            correct += 1
        
#    print(correct/feats.shape[0])

def solve_sudoku(game):
    
    #game = game.replace('\n', '')
    #game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


def printing(arr, truth):
    stri = ""
    for i in range(9):
        for j in range(9):
            stri += str(arr[i][j])
    if stri == truth:
        return 1
    return 0


def check_sudoku(grid):
    if len(grid) == 9:
        numsinrow = 0
        for i in range(9):
            if len(grid[i]) == 9:
                numsinrow += 1
        if numsinrow == 9:
            for i in range(9):
                rowoccurence = [0,0,0,0,0,0,0,0,0,0]
                for j in range(9):
                    rowoccurence[grid[i][j]] += 1
                    temprow = rowoccurence[1:10]
                    if temprow == [1,1,1,1,1,1,1,1,1]:
                        return True
                    else:
                        return False
        else:
            return False
    else:
        return False



nb=10
verif=0

start = time.time()

for n in range(len(data)-1,len(data)-1-nb,-1):

    game = data.values[n][0]
    game = solve_sudoku(game)

    verif+=printing(game,data.values[n][1])
    
end = time.time()

print(f"Sample of {nb} SUDOKUS")
print("Score = ", int(verif/nb*100))
print("Duration = ", round(end - start, 2), "s")