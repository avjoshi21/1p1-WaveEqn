import numpy as np
from matplotlib import pyplot as plt
import copy
import time


#second order derivative calculator. returns entire array with derivative at each point.
def spatialDerivCalc(variableArray,delta,order=2):
    if order==2:
        paddedArray = np.pad(variableArray,1,mode='wrap')
        paddedArray = np.pad(variableArray,1,mode='wrap')
        fDiffArray = np.diff(paddedArray[1:])
        bDiffArray = np.diff(paddedArray[::-1][1:])[::-1]
        spatialDerivArray = (fDiffArray - bDiffArray)/(2*delta)

        ## Slower code that does the same thing as above
        # for count,i in enumerate(variableArray):
        #     paddedCount = count+1
        #     prevVal = paddedArray[paddedCount-1]
        #     nextVal = paddedArray[paddedCount+1]
        #     # currVal = i
        #     spatialDerivArray[count] = (nextVal - prevVal)/(2*delta)


    elif order==4:
        paddedArray = np.pad(variableArray,2,mode='wrap')
        spatialDerivArray = np.zeros(variableArray.shape)
        for count,i in enumerate(variableArray):
            paddedCount = count+2
            # p1=previous1st n2=next2nd
            p1Val = paddedArray[paddedCount-1]
            n1Val = paddedArray[paddedCount+1]
            p2Val = paddedArray[paddedCount-2]
            n2Val = paddedArray[paddedCount+2]
            # currVal = i
            spatialDerivArray[count] = (-n2Val + 8*n1Val- 8*p1Val + p2Val)/(12* delta)
    
    return spatialDerivArray

def piPsiPhiRK4(pi,psi,phi,deltaT,deltaX):

    k1Pi = spatialDerivCalc(psi,deltaX,order=4)
    k1Psi = spatialDerivCalc(pi,deltaX,order=4)
    k1Phi = pi

    k2Pi = spatialDerivCalc(psi + deltaT/2 * k1Psi,deltaX,order=4)
    k2Psi = spatialDerivCalc(pi + deltaT/2 * k1Pi,deltaX,order=4)
    k2Phi = pi + deltaT/2 * k1Pi

    k3Pi = spatialDerivCalc(psi + deltaT/2 * k2Psi,deltaX,order=4)
    k3Psi = spatialDerivCalc(pi + deltaT/2 * k2Pi,deltaX,order=4)
    k3Phi = pi + deltaT/2 * k2Pi

    k4Pi = spatialDerivCalc(psi + deltaT * k3Psi,deltaX,order=4)
    k4Psi = spatialDerivCalc(pi + deltaT * k3Pi,deltaX,order=4)
    k4Phi = pi + deltaT * k3Pi

    finalPi = pi + 1/6*deltaT*(k1Pi + 2*k2Pi + 2*k3Pi + k4Pi)
    finalPsi = psi + 1/6*deltaT*(k1Psi + 2*k2Psi + 2*k3Psi + k4Psi)
    finalPhi = phi + 1/6*deltaT*(k1Phi + 2*k2Phi + 2*k3Phi + k4Phi)

    return finalPi,finalPsi,finalPhi


if __name__ == "__main__":
    #grid setup
    hx = 0.01
    ht = 0.0025
    xLow = -2
    xHigh = 2
    tLow = 0
    tHigh = 5
    xGrid = np.arange(xLow,xHigh+hx,hx)
    tGrid = np.arange(tLow,tHigh,ht)

    #initial values of the field phi and simulation variables psi and pi
    initialPhi = np.exp(-((xGrid-0.5)/0.5)**2)
    initialDtPhi = np.zeros(xGrid.shape)
    initialDxPhi = spatialDerivCalc(initialPhi,hx,order=4)

    initialPsi = initialDxPhi
    initialPi = initialDtPhi

    #set the arrays (t and x) of phi, psi and pi
    phiArray = np.zeros((len(tGrid)+1,len(xGrid)))
    psiArray = np.zeros((len(tGrid)+1,len(xGrid)))
    piArray = np.zeros((len(tGrid)+1,len(xGrid)))


    phiArray[0] = initialPhi
    psiArray[0] = initialPsi
    piArray[0] = initialPi

    t1 = time.perf_counter()
    for count,tVal in enumerate(tGrid):
        piArray[count+1],psiArray[count+1],phiArray[count+1] = piPsiPhiRK4(piArray[count],psiArray[count],phiArray[count],ht,hx)

    t2 = time.perf_counter()
    print('Integration finished in {:.4f} seconds. Now plotting...............'.format(t2-t1))
    for count,psiVals in enumerate(phiArray[::5]):
        fig = plt.figure(count)
        ax = plt.gca()
        plt.xlabel('x')
        plt.ylabel('Phi')
        ax.set_ylim([-np.max(phiArray),np.max(phiArray)])
        plt.plot(xGrid,psiVals,label="t={:.2f}".format(tGrid[count]))
        plt.legend()
        plt.savefig('Images/frame_{:04d}.png'.format(count))
        plt.close(fig)
