import numpy as np
from matplotlib import pyplot as plt
import copy

#finds and returns index and value of the array element closest to parameter
def findNearest(array, value,startIdx=0):
    array = np.asarray(array)
    bestValue = abs(array[startIdx] - value)
    # print(startIdx)
    bestIdx = startIdx
    prevIdx = startIdx-1
    nextIdx = startIdx+1
    # print(prevIdx)
    while(prevIdx >=0 or nextIdx<len(array)):
        if(prevIdx>=0):
            prevVal = array[prevIdx]
            if(abs(prevVal-value) < bestValue):
                # print('found better one')
                bestValue = abs(prevVal-value)
                bestIdx = prevIdx;
            prevIdx -= 1
        if(nextIdx<len(array)):
            nextVal = array[nextIdx]
            if(abs(nextVal-value) < bestValue):
                bestValue = abs(nextVal-value)
                bestIdx = nextIdx;
            nextIdx += 1

    return (bestIdx,array[bestIdx])

#second order derivative calculator. returns entire array with derivative at each point.
def spatialDerivCalc(variableArray,delta):
    spatialDerivArray = np.zeros(variableArray.shape)
    prevVal=0
    # currVal=0
    nextVal=0
    for count,i in enumerate(variableArray):
        if (count==0):
            prevVal = variableArray[-1]
        else:
            prevVal = variableArray[count-1]
        if (count==len(variableArray)-1):
            nextVal = variableArray[0]
        else:
            nextVal = variableArray[count+1]
        # currVal = i
        spatialDerivArray[count] = (nextVal - prevVal)/(2*delta)
    return spatialDerivArray

#performs a single rk4 step for an array of values. diffArrayFunc is a function passed that
#computes the derivative (discretized or continuous) by accepting the grid and the target value it should check

def rk4(initialArray,gridArray,delta,diffArrayFunc,**diffArrayFuncArgs):
    # finalArray = np.zeros(len(initialArray))
    finalArray = copy.copy(initialArray)
    for count,val in enumerate(initialArray):
        y0 = val
        k1 = diffArrayFunc(count,**diffArrayFuncArgs)
        y1 = (y0 + delta/2 * k1)
        y1Idx = findNearest(gridArray,y1)[0]
        # y1Idx = findNearest(initialArray,y1,count)[0]
        k2 = diffArrayFunc(y1Idx,**diffArrayFuncArgs)
        y2 = (y0 + delta/2 * k2)
        y2Idx = findNearest(gridArray,y2)[0]
        # y2Idx = findNearest(initialArray,y2,y1Idx)[0]
        k3 = diffArrayFunc(y2Idx,**diffArrayFuncArgs)
        y3 = (y0 + delta * k3)
        y3Idx = findNearest(gridArray,y3)[0]
        # y3Idx = findNearest(initialArray,y3,y2Idx)[0]
        k4 = diffArrayFunc(y3Idx,**diffArrayFuncArgs)
        y4 = y0 + (k1 + 2*k2+ 2*k3 + k4)*delta/6
        finalArray[count] = y0 + delta*k1
        # finalArray[count] = y4
    return finalArray
        


#grid setup
hx = 0.1
ht = 0.05
xLow = -5
xHigh = 5
tLow = 0
tHigh = 10
xGrid = np.arange(xLow,xHigh+hx,hx)
tGrid = np.arange(tLow,tHigh,ht)

#initial values of the field phi and simulation variables psi and pi
initialPhi = np.exp(-(xGrid)**2)
initialDtPhi = np.zeros(xGrid.shape)
initialDxPhi = spatialDerivCalc(initialPhi,hx)

initialPsi = initialDxPhi
initialPi = initialDtPhi

#set the arrays (t and x) of phi, psi and pi
phiArray = np.zeros((len(tGrid)+1,len(xGrid)))
psiArray = np.zeros((len(tGrid)+1,len(xGrid)))
piArray = np.zeros((len(tGrid)+1,len(xGrid)))


phiArray[0] = initialPhi
psiArray[0] = initialPsi
piArray[0] = initialPi

def derivCalc(idx,**kwargs):
    #if type is Psi, send the value of time derivative of psi, which is the spatial derivative of pi and
    #vice versa.
    if(kwargs['type']=='Pi'):
        return kwargs['dxPsi'][idx]
    elif(kwargs['type']=='Psi'):
        return kwargs['dxPi'][idx]
    elif(kwargs['type']=='Phi'):
        return kwargs['piArray'][idx]

for count,tVal in enumerate(tGrid):
    dxPsi = spatialDerivCalc(psiArray[count],hx)
    dxPi = spatialDerivCalc(piArray[count],hx)
    # if count == 1:
    #     plt.plot(xGrid,dxPi)
    
    piArray[count+1] = rk4(piArray[count],xGrid,ht,derivCalc,type='Pi',dxPsi=dxPsi)
    psiArray[count+1] = rk4 (psiArray[count],xGrid,ht,derivCalc,type='Psi',dxPi=dxPi)
    # print(count)
    phiArray[count+1] = rk4(phiArray[count],xGrid,ht,derivCalc,type='Phi',piArray=piArray[count])

for count,psiVals in enumerate(phiArray):
    fig = plt.figure(count)
    ax = plt.gca()
    plt.xlabel('x')
    plt.ylabel('Phi')
    ax.set_ylim([0,np.max(phiArray)])
    plt.plot(xGrid,psiVals,label="t="+str(tGrid[count]))
    plt.legend()
    plt.savefig('Images/frame_%04d.png' %count)
    plt.close(fig)
# plt.show()

# y0=3.0
# hVals = [0.8,0.2,0.1,0.02,0.01]
# for h in hVals:
#     trange = np.arange(0.0,2.0,h)
#     ytrue = lambda tVals: 3 * np.exp(-2 * tVals)
#     yVals = np.zeros(len(trange)+1,dtype=float)
#     yGrid = np.linspace(0,2,10000)
#     yVals[0]=3.0
#     def diffFunc(y,yVal):
#         # idx = findNearest(y,yVal)
#         return -2.0 * yVal
#     for count,i in enumerate(trange):
#         yVals[count+1] = rk4([yVals[count]],yGrid,diffFunc,h)

#     plt.plot(trange,yVals[:-1],'x-',label="h= "+str(h))
# plt.plot(np.linspace(0,2,1000),ytrue(np.linspace(0,2,1000)),label="true solution")
# plt.legend()
# plt.show()


