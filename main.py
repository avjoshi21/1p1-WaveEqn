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

def piPsiPhiRK4(pi,psi,phi,deltaT,deltaX,order=2):

    k1Pi = spatialDerivCalc(psi,deltaX,order)
    k1Psi = spatialDerivCalc(pi,deltaX,order)
    k1Phi = pi

    k2Pi = spatialDerivCalc(psi + deltaT/2 * k1Psi,deltaX,order)
    k2Psi = spatialDerivCalc(pi + deltaT/2 * k1Pi,deltaX,order)
    k2Phi = pi + deltaT/2 * k1Pi

    k3Pi = spatialDerivCalc(psi + deltaT/2 * k2Psi,deltaX,order)
    k3Psi = spatialDerivCalc(pi + deltaT/2 * k2Pi,deltaX,order)
    k3Phi = pi + deltaT/2 * k2Pi

    k4Pi = spatialDerivCalc(psi + deltaT * k3Psi,deltaX,order)
    k4Psi = spatialDerivCalc(pi + deltaT * k3Pi,deltaX,order)
    k4Phi = pi + deltaT * k3Pi

    finalPi = pi + 1/6*deltaT*(k1Pi + 2*k2Pi + 2*k3Pi + k4Pi)
    finalPsi = psi + 1/6*deltaT*(k1Psi + 2*k2Psi + 2*k3Psi + k4Psi)
    finalPhi = phi + 1/6*deltaT*(k1Phi + 2*k2Phi + 2*k3Phi + k4Phi)

    return finalPi,finalPsi,finalPhi


def runSimulation(**simPars):
        #grid setup
    # hx = 0.01
    # ht = 0.0025
    # xLow = -2
    # xHigh = 2
    # tLow = 0
    # tHigh = 5

    xGrid = np.arange(simPars['xLow'],simPars['xHigh'],simPars['hx'])
    tGrid = np.arange(simPars['tLow'],simPars['tHigh'],simPars['ht'])

    #initial values of the field phi and simulation variables psi and pi
    initialPhi = np.exp(-((xGrid-0.5)/0.5)**2)
    initialDtPhi = np.zeros(xGrid.shape)
    initialDxPhi = spatialDerivCalc(initialPhi,simPars['hx'],order=simPars['spatialDerivOrder'])

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
        piArray[count+1],psiArray[count+1],phiArray[count+1] = piPsiPhiRK4(piArray[count],psiArray[count],phiArray[count],simPars['ht'],simPars['hx'],simPars['spatialDerivOrder'])

    t2 = time.perf_counter()
    print('Integration finished in {:.4f} seconds.'.format(t2-t1))

    if(simPars['toPlot']):
        print('Plotting......')
        skipVal = max(len(phiArray)//(400),1)
        for count,psiVals in enumerate(phiArray[::skipVal]):
            fig = plt.figure(count)
            ax = plt.gca()
            plt.xlabel('x')
            plt.ylabel(r'$\Phi$')
            ax.set_ylim([-np.max(phiArray),np.max(phiArray)])
            plt.plot(xGrid,psiVals,label="t={:.2f}".format(tGrid[count*skipVal]))
            plt.legend()
            plt.savefig('Images/frame_{:04d}.png'.format(count))
            plt.close(fig)


    return phiArray

def convergenceTest(**simPars):
    if(simPars['convergenceOfT']):
        # xGrid = np.arange(simPars['xLow'],simPars['xHigh'],simPars['hx'])

        # truePhi = np.exp(-((xGrid-0.5)/0.5)**2)

        htValues = [0.01,0.005,0.0025,0.00125]
        phiSolutions = []
        # errorsat4 = []
        
        for htVal in htValues:
            simPars['ht']=htVal
            # tGrid = np.arange(simPars['tLow'],simPars['tHigh'],simPars['ht'])

            phiArr = runSimulation(**simPars)
            sol = phiArr[-1]
            phiSolutions.append(sol)
        
        L1NormDifference = np.sum(np.abs(np.diff(phiSolutions,axis=0)),axis=-1)
        print(L1NormDifference)
        # return phiSolutions
        convergenceFactor = []
        for i in range(len(L1NormDifference)-1):
            convergenceFactor.append(L1NormDifference[i]/L1NormDifference[i+1])

        return convergenceFactor

        # fig,ax = plt.subplots(nrows=2,ncols=1)
        # for count,i in enumerate(solutionsAt4):
        #     ax[0].plot(xGrid,i,label="ht = {}".format(htValues[count]))
        # ax[0].legend()
        # ax[1].loglog(htValues,errorsat4)
        # plt.show()

if __name__ == "__main__":

    simPars = {
        'hx':0.01,'ht':0.025,
        'xLow':-2,'xHigh':2,
        'tLow':0,'tHigh':10,
        'toPlot':False,
        'spatialDerivOrder':2
        }

    cFs = convergenceTest(**simPars,convergenceOfT=True)
 
    # hx = 0.01
    # ht = 0.0025
    # xLow = -2
    # xHigh = 2
    # tLow = 0
    # tHigh = 5
    
