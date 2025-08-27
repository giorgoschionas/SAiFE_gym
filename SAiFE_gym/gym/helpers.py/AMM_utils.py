import numpy as np

def CPMM_Spot_Price (X, Y):
    return X / Y

def update_CPMM(X, Y, dX, dY):
    X += dX
    Y += dY
    return X, Y

def Sell_CPMM(sell, X, Y, eta):
    output = X - (X * Y) / (Y + sell * (1 - eta))
    return output

def Buy_CPMM(buy, X, Y, eta):
    output = Y - (Y * X) / (X + buy * (1 - eta))
    return output

def Arb_Trade_CPMM(S, X, Y, eta):
    sell_output = Sell_CPMM(S, X, Y, eta)
    buy_output = Buy_CPMM(S, X, Y, eta)
    return sell_output, buy_output