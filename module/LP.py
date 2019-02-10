# date : 2/11/2019
# author : takeshi
import pandas as pd
import numpy as np
from IPython.display import display

def linprog(c,A,comp,b,maximize=True):
    '''
    Maximize(or Minimize) a linear objective function subject to linear equality and inequality constraints.

    Linear Programming is intended to solve the following problem form:

    Maximize: c * x

    Subject to: A * x [comp] b , (x >= 0)

    Parameters
    ----------
    c : array_like
        Coefficients of the linear objective function to be maximized.
    A : array_like
        2-D array which, when matrix-multiplied by x, 
        gives the values of constraints at x.
    comp : array_like
        1-D array of values representing a sign of equality in each constraint (row).
        if value is -1 , it means (<=)
        if value is 0 , it means (=)
        if value is 1 , it means (=>)
    b : array_like
        1-D array of values representing the RHS of each constraint (row).

    maximize : bool, optional
        If True, the linear objective function is to be maximized.
        If False, the linear objective function is to be minimized.
        (the default is True)
        
    Returns
    -------
    pandas.DataFrame
        final simplex table.  
        Optimal solution is table['Values'] , and Optimal value is table['z','Values'].
        if x is (1 * n) matrix , x_i ( i >= n ) is Slack Variable.
    '''

    # optimize
    def optimize(table,target):
        if not __debug__:
            if target == 'w':
                print("Phase 1 : find initial solution")
            else:
                if maximize:
                    print("Phase 2 : Maximize the liner objective function")
                else:
                    print("Phase 2 : Minimize the liner objective function")
        baseIndex = table.index.values
        nonBaseIndex = np.setdiff1d(np.vectorize(lambda i : 'x' + str(i))(np.arange(len(table.columns)-1)) ,baseIndex)
        for i in range(100000):
            if not __debug__:
                print("roop {0}".foramt(i))
                display(table)
            nonBaseTable = table.loc[target,nonBaseIndex]
            if ((nonBaseTable < -1e-8).values.sum()) == 0:
                return table
                # 新たな基底変数
            nextIndex = (nonBaseTable.map(lambda x: -x)).idxmax(axis=1)
            # 取り替えられる基底変数
            idx = table.index.get_loc(target)
            tmpLine = (table['Value'].iloc[:idx] / table.loc[ : ,nextIndex].iloc[:idx] )
            prevIndex = str(tmpLine.map(lambda x: float('inf') if x < 0 else x ).idxmin())
            nonBaseIndex[np.where(nonBaseIndex == nextIndex)] = prevIndex
            table = table.rename(index={prevIndex : nextIndex})
            table.loc[nextIndex] /= table.at[nextIndex,nextIndex]
            pivotLine = table.loc[nextIndex]
            unPivotIndex = list(table.index.drop(nextIndex))
            table.loc[unPivotIndex] = table.loc[unPivotIndex].apply(lambda x: x - (x.at[nextIndex]*pivotLine) ,axis=1)

        print("cannot find base solutions")

    if not maximize: 
        c = (-c)
    n,m = A.shape
    slackVariableNum = 0
    artificialVariableNum = 0
    slackVariable = [0] * n
    artificialVariable = [0] * n
    for i in range(n):
        # bの値を全て正の値にしておく
        if b[i] < 0:
            A[i] = -A[i]
            comp[i] = -comp[i]
            b[i] = -b[i]
        # < ( -> スラック変数を導入 )
        if comp[i] == -1:
            slackVariableNum += 1
            slackVariable[i] = 1
        # = ( -> 人為変数を導入 )
        elif comp[i] == 0:
            artificialVariableNum += 1
            artificialVariable[i] = 1
        # > ( -> スラック変数,人為変数を導入 )
        else:
            slackVariableNum += 1
            artificialVariableNum += 1
            slackVariable[i] = -1
            artificialVariable[i] = 1

    variableNum = c.shape[0] + slackVariableNum + artificialVariableNum
    addVariableNum =  slackVariableNum + artificialVariableNum

    # Valueを求める.
    baseIndex = np.empty(n)
    baseValue = np.empty(n)
    A_ = np.append(A , np.zeros((n,addVariableNum)),axis=1)
    slackIter = c.shape[0] 
    artificialIter = c.shape[0] + slackVariableNum

    # (スラック変数 < 人為変数) の優先順位で基底変数に選ぶ.
    # すると , i 本目の制約条件式のみに登場する変数を選ぶことができる.
    # baseIndex[i] := i 本目の制約条件式のみに登場する変数の番号
    # baseValue[i] := i本目の制約条件式のみに登場する変数の値 ( = Value = b[i] ) となる.
    for i in range(n):
        if slackVariable[i] != 0:
            A_[i,slackIter] = slackVariable[i]
            # 1の場合
            if slackVariable[i] > 0:
                baseIndex[i],baseValue[i] = slackIter, b[i]
            slackIter += 1
            
        if artificialVariable[i] != 0:
            A_[i,artificialIter] = artificialVariable[i]
            baseIndex[i],baseValue[i] = artificialIter, b[i]
            artificialIter += 1 

    # フェーズ1 (Valueを見つける)
    # 目的関数の値をzとおく
    # Valueの列を追加
    exA = np.append(baseValue.reshape(n,1),A_,axis=1)
    # zの行を追加
    c_ = np.array([0]*(c.shape[0] + slackVariableNum) + [-1]*(artificialVariableNum))
    c_ = c_[np.vectorize(int)(baseIndex)]
    w = (c_ @ exA).reshape(1,variableNum+1)
    z = np.append(np.append(np.zeros(1),-c),np.array([0]*addVariableNum)).reshape(1,variableNum+1)
    table = np.append(np.append(exA,w,axis=0),z,axis=0)
    # データフレームにする
    df = pd.DataFrame(table,
        columns=['Value']+[ 'x' + str(i) for i in range(variableNum)],
        index= list(np.vectorize(lambda i: 'x' + str(int(i)))(baseIndex))  + ['w','z']
    )
    table = optimize(df,'w')
    if artificialVariableNum != 0:
        table = table.iloc[:,:-artificialVariableNum]
    variableNum -= artificialVariableNum
    table = table.drop('w')
    result = optimize(table,'z')
    if not maximize:
        result['Value']['z'] = -result['Value']['z']
    return result

## Example
if __name__ == '__main__':
    # maximize  2 * x_0 + 3 * x_1
    # constraints : 
    #  1 * x_0 + 2 * x_1 <= 10
    #  2 * x_0 + 1 * x_0 <= 8
    #  ( x_0 >= 0 , x_1 >= 0)

    c = np.array([ 2,3])
    A = np.array([  [1,2],
                    [2,1] ])
    comp = np.array([-1,-1])
    b = np.array([10,8])

    # solve
    df = linprog(c,A,comp,b,True)
    # result
    print(df)
    