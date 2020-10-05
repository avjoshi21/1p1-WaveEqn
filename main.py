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
                bestIdx = prevIdx
            prevIdx -= 1
        if(nextIdx<len(array)):
            nextVal = array[nextIdx]
            if(abs(nextVal-value) < bestValue):
                bestValue = abs(nextVal-value)
                bestIdx = nextIdx
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

def piPsiPhiRK4(pi,psi,phi,deltaT,deltaX):
    finalPi = copy.deepcopy(pi)
    finalPsi = copy.deepcopy(psi)
    finalPhi = copy.deepcopy(phi)

    k1Pi = spatialDerivCalc(psi,deltaX)
    k1Psi = spatialDerivCalc(pi,deltaX)
    k1Phi = pi

    k2Pi = spatialDerivCalc(psi + deltaT/2 * k1Psi,deltaX)
    k2Psi = spatialDerivCalc(pi + deltaT/2 * k1Pi,deltaX)
    k2Phi = pi + deltaT/2 * k1Pi

    k3Pi = spatialDerivCalc(psi + deltaT/2 * k2Psi,deltaX)
    k3Psi = spatialDerivCalc(pi + deltaT/2 * k2Pi,deltaX)
    k3Phi = pi + deltaT/2 * k2Pi

    k4Pi = spatialDerivCalc(psi + deltaT * k3Psi,deltaX)
    k4Psi = spatialDerivCalc(pi + deltaT * k3Pi,deltaX)
    k4Phi = pi + deltaT * k3Pi

    finalPi = finalPi + 1/6*deltaT*(k1Pi + 2*k2Pi + 2*k3Pi + k4Pi)
    finalPsi = finalPsi + 1/6*deltaT*(k1Psi + 2*k2Psi + 2*k3Psi + k4Psi)
    finalPhi = finalPhi + 1/6*deltaT*(k1Phi + 2*k2Phi + 2*k3Phi + k4Phi)

    return finalPi,finalPsi,finalPhi


#performs a single rk4 step for an array of values. diffArrayFunc is a function passed that
#computes the derivative (discretized or continuous) by accepting the grid and the target value it should check

def rk4(initialArray,gridArray,delta,diffArrayFunc,**diffArrayFuncArgs):
    # finalArray = np.zeros(len(initialArray))
    finalArray = copy.copy(initialArray)
    for count,val in enumerate(initialArray):
        y0 = val
        k1 = diffArrayFunc(count,**diffArrayFuncArgs)
        y1 = (y0 + delta/2 * k1)
        # y1Idx = findNearest(gridArray,y1)[0]
        y1Idx = findNearest(initialArray,y1,count)[0]
        k2 = diffArrayFunc(y1Idx,**diffArrayFuncArgs)
        y2 = (y0 + delta/2 * k2)
        # y2Idx = findNearest(gridArray,y2)[0]
        y2Idx = findNearest(initialArray,y2,y1Idx)[0]
        k3 = diffArrayFunc(y2Idx,**diffArrayFuncArgs)
        y3 = (y0 + delta * k3)
        # y3Idx = findNearest(gridArray,y3)[0]
        y3Idx = findNearest(initialArray,y3,y2Idx)[0]
        k4 = diffArrayFunc(y3Idx,**diffArrayFuncArgs)
        y4 = y0 + (k1 + 2*k2+ 2*k3 + k4)*delta/6
        finalArray[count] = y0 + delta*k1
        # finalArray[count] = y4
    return finalArray
        


#grid setup
hx = 0.1
ht = 0.01
xLow = -5
xHigh = 5
tLow = 0
tHigh = 5
xGrid = np.arange(xLow,xHigh+hx,hx)
tGrid = np.arange(tLow,tHigh,ht)

#initial values of the field phi and simulation variables psi and pi
initialPhi = np.exp(-((xGrid-1)/0.5)**2)
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
    # dxPsi = spatialDerivCalc(psiArray[count],hx)
    # dxPi = spatialDerivCalc(piArray[count],hx)
    # if count == 1:
    #     plt.plot(xGrid,dxPi)
    
    piArray[count+1],psiArray[count+1],phiArray[count+1] = piPsiPhiRK4(piArray[count],psiArray[count],phiArray[count],ht,hx)
    # phiArray[count+1] = phiRK4(phiArray[count],piArray[count],ht,hx)
    # piArray[count+1] = rk4(piArray[count],xGrid,ht,derivCalc,type='Pi',dxPsi=dxPsi)
    # psiArray[count+1] = rk4 (psiArray[count],xGrid,ht,derivCalc,type='Psi',dxPi=dxPi)
    # # print(count)
    # phiArray[count+1] = rk4(phiArray[count],xGrid,ht,derivCalc,type='Phi',piArray=piArray[count])

print('Integration finished. Now plotting...............')
for count,psiVals in enumerate(phiArray):
    fig = plt.figure(count)
    ax = plt.gca()
    plt.xlabel('x')
    plt.ylabel('Phi')
    ax.set_ylim([-np.max(phiArray),np.max(phiArray)])
    plt.plot(xGrid,psiVals,label="t={:.2f}".format(tGrid[count]))
    plt.legend()
    plt.savefig('Images/frame_{:04d}.png'.format(count))
    plt.close(fig)



