import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import scipy.optimize as sc
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go


stockList = [str(x) for x in input("Enter Stocks You want to Compare: ").split()]

#stockList = ['ITC','RELIANCE','LT']
stocks = [stock + '.NS' for stock in stockList]



def getData(stocks, start, end):
    stockData=pd.DataFrame()
    for stock in stocks:
        data = yf.download(stock,start=start,end=end)
        data=data['Close']
        stockData = pd.concat([stockData, data],axis=1)
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std=np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std


def negativeSR(weights, meanReturns, covMatrix, riskFreeRate):
    pReturns, pStd = portfolioPerformance(weights,meanReturns,covMatrix)
    return -(pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq','fun': lambda x : np.sum(x) -1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args, method='SLSQP',bounds=bounds, constraints=constraints)
    return result 

def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]
    
def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    "Minimize the portfolio variance by altering the weights/allocation of assets in the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq','fun': lambda x : np.sum(x) -1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method='SLSQP',bounds=bounds, constraints=constraints)

    return result 



endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)


meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)

def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]

def efficientOptim(meanReturns, covMatrix, returnTarget, constraintSet =(0,1)):
    """For Each Return Target, we want to optimix=ze the piortfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns,covMatrix)

    constraints = ({'type':'eq','fun':lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                    {'type':'eq','fun': lambda x : np.sum(x) -1})
    bound = constraintSet
    bounds = (constraintSet for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance,numAssets*[1./numAssets],args=args,method='SLSQP',bounds=bounds,constraints=constraints)
    return effOpt



def calculatedResults(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    """Read in mean,cov Matrix, and other financial information
    Output, MaxSR, Min Volatility, efficient frontier"""
    #MAX SHARPE RATIO PORTFOLIO
    maxSR_Portfolio = maxSR(meanReturns,covMatrix)
    maxSR_returns , maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'],index=meanReturns.index,columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]

    #MIN VOLATILITY PORTFOLIO
    minVol_Portfolio = minimizeVariance(meanReturns,covMatrix)
    minVol_returns , minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'],index=meanReturns.index,columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]
    

    #EFFICIENT FRONTIER
    efficientList = []
    targetReturns = np.linspace(minVol_returns,maxSR_returns, 20)

    for target in targetReturns:
       efficientList.append(efficientOptim(meanReturns,covMatrix,target)['fun'])

    maxSR_returns , maxSR_std = round(maxSR_returns*100,2) , round(maxSR_std*100,2)
    minVol_returns , minVol_std = round(minVol_returns*100,2) , round(minVol_std*100,2)
    
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation,efficientList, targetReturns


#print(calculatedResults(meanReturns,covMatrix))

def ef_graph(meanReturns, covMatrix, riskFreeRate =0,constraintSet=(0,1)):
     """Return a graph plotting the min vol, max sr and efficient frontier"""
     maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation,efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)

     #MAX SR
     MaxSharpeRatio = go.Scatter(
        name="Maximum Sharpe Ratio",
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red',size=14,line=dict(width=3,color='black'))
     )

     #MIN SR
     MinVol = go.Scatter(
        name="Minimum Volatility",
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
        marker=dict(color='green',size=14,line=dict(width=3,color='black'))
     )

    #EFFICIENT FRONTIER
     EF_curve = go.Scatter(
        name="Efficient Frontier",
        mode='lines',
        x=[round(ef_std*100,2) for ef_std in efficientList],
        y=[round(target*100,2) for target in targetReturns],
        line=dict(color='black',width=4,dash='dashdot')
     )

     data = [MaxSharpeRatio,MinVol,EF_curve]

     layout = go.Layout(
        title = "Portfolio Optimization with the Efficient Frontier",
        yaxis = dict(title='Annualized Return (%)'),
        xaxis = dict(title='Annualized volatiltiy (%)'),
        showlegend = True,
        legend = dict(
            x=0.75, y=0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=1500,
        height=700,)

     fig = go.Figure(data=data, layout=layout)
     return fig.show()

ef_graph(meanReturns,covMatrix)


