# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:35:33 2019

this module is for targetn computations

the methods below do not implement admission control or
driver rebalancing, they are used to compute the targets only
refer to manage-demand-adm.py for the full framework

@author: cesny
"""
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import scipy.stats as st
from utils import readCSV, addTimeStamp, computeNumberOfUsers, getNumTrips, avgNumPerWindow, retrieveEntries, generateBA, maintainInOut, InOutFromDict, sumDicts, diffDicts, getPortionDict, map2time
from rates import getOrderedService, checkPoisson, getMu, getLambdaMLE, getMuPerRegion, getLamPerRegionMLE, getArrivalRates, getArrivalRatesSingle, getOrderedServiceSingle


# ~~ empirical distribution ~~~~~~
def getEmpiricalIntegral(t, listOS):
    '''
    ---------------------------------------------------------------------------
    Gets the integration of the CDF of the empirical distribution up to time t
    i.e., computes int_{0}^{t}P(S<=u)du
    ---------------------------------------------------------------------------
    t: time point
    listOS: list of ordered service times obtained from getOrderedService
    the service times in the list must be in units of slots
    ---------------------------------------------------------------------------
    '''
    # get number of data points
    n = len(listOS)
    # first get the list of all the service times that are less than t in the window of interest
    storeService = list()
    key=0
    NotComplete = True
    while (NotComplete) and (listOS[key] <= t):
        storeService.append(listOS[key])
        key+=1
        if key not in np.arange(0, len(listOS), 1):  # check if index out of range
            NotComplete = False  # i.e., complete
    # add the time point as the last point
    storeService.append(t)
    # get empirical distribution CDF step function up to time t
    x = list()
    x.append(0)
    for elem in storeService[:-1]:  # for all elements up until the last one store them twice to make the step function
        x.extend([elem]*2)
    x.append(storeService[-1])  # add the last element
    y = list()
    y.append(0)  # first number is (zero, zero)
    cuml = 0  # stores cumulative
    for elem in storeService[:-1]:
        y.append(cuml)
        cuml += 1.0/n
        y.append(cuml)
    y.append(cuml)  # at time t you will have the last cumulative count
    # now do the integration
    integral = np.trapz(y,x)
        
    return integral


def getRhoTEmp(Lambda, listOS, timePt, window_length, window_end_slot):
    '''
    ----------------------------------------------------------------------------------
    retrieves the time-varying poisson rate at the time point for an M/GI/infty
    queue that starts empty at the beginning of the window
    ----------------------------------------------------------------------------------
    timePt is in units of time point
    listOS: list of ordered service times obtained getOrderedService for time window
    of interest
    ----------------------------------------------------------------------------------
    Lambda is in units of slots and averaged over time window
    
    The Lambda must be thinned before entered here:
        Lambda of the stochastic is (1-Pba)*LambdaSlots where 
        Lambda Slots is calibrated from data and in units of slots
    Note: computations are at time points 
    instead of time slots, t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    ----------------------------------------------------------------------------------
    '''
    time = window_length-(window_end_slot - timePt)  # change of variables: re-zeroes time based on the window
    rhoT = Lambda*time - Lambda*getEmpiricalIntegral(time, listOS)
    return rhoT

def computeCDFEmp(Lambda, listOS, timePt, window_length, window_end_slot, bookAheads, n):
    '''
    ---------------------------------------------------------------------------------------------------
    at a specific time point, given arrival rate, service rate
    number of bookaheads, and target n, we compute
    P(S(t)<c-b(t)) where:
    - S(t) is R.V. instantaneous number of users in system (num. of busy servers in M/GI/infty queue)
    - b(t) is bookaheads at time point (known) = fBA + fP (book-ahead and initiated in prev. window)
    - c is target (referred to as n above)
    ---------------------------------------------------------------------------------------------------
    Note: arrival rate should be of instantaneous only (thin arrival rate based on percent BA)
    Note: lambda must be given per time slot!
    Note: computations are at time points instead of time slots
    time points: t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    ---------------------------------------------------------------------------------------------------
    '''
    rhoT = getRhoTEmp(Lambda, listOS, timePt, window_length, window_end_slot)
    maxBA = bookAheads[timePt]
    for tP in np.arange(timePt, window_end_slot+1, 1):
        if bookAheads[tP] > maxBA:
            maxBA=bookAheads[tP]
    CDF = st.poisson.cdf(n-maxBA, rhoT)  # get the CDF of a poisson with paramater rhoT
    return CDF

def computeProbEmp(Lambda, listOS, timePt, window_length, window_end_slot, bookAheads, n):
    '''
    ------------------------------------------------------------------------------------------------------
    at a specific time slot, given arrival rate, service rate
    number of bookaheads, and target n, we compute
    P(S(t)>c-b(t)) where:
        - S(t) is R.V. instantaneous number of users in system (num. of busy servers in M/GI/infty queue)
        - b(t) is bookaheads at time slot (known) = fBA + fP (book-ahead and initiated in prev. window)
        - c is target (referred to as n above)
    ------------------------------------------------------------------------------------------------------
    Note: arrival rates should be of instantaneous only (thin the Poisson process)
    Note: lambda and mu must be given per time slot!
    Note: computations are at time points  
    instead of time slots, t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    ------------------------------------------------------------------------------------------------------
    '''
    Prob = 1 - computeCDFEmp(Lambda, listOS, timePt, window_length, window_end_slot, bookAheads, n)
    return Prob

def getSampleProfileEmp(lamWin, listOS, window_length, windowFirstSlot, percentBooked):
    '''
    ------------------------------------------------------------------------------------------------------
    for a specific window (as determined by windowFirstSlot and lamWin) generate the expected number
    of instantaneous users at every time point, this corresponds to the mean of the time-dependent R.V.
    representing the num. of users in an M/GI/infty queue
    ------------------------------------------------------------------------------------------------------
    NOTE: THIS FUNCTION THINS LAMBDA BASED ON PERCENT BOOKED
    ------------------------------------------------------------------------------------------------------
    Note: computations are at time points 
    instead of time slots, t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    ------------------------------------------------------------------------------------------------------
    '''
    thinnedLambda = lamWin * (1 - percentBooked)  # this gives you the lambda for instantaneous
    windowEndSlot = windowFirstSlot + window_length - 1
    sampleProf = dict()  # store the sampled profile
    for tp in np.arange(windowFirstSlot, windowEndSlot+1, 1):
        rho = getRhoTEmp(Lambda=thinnedLambda, listOS=listOS, timePt=tp, window_length=window_length, window_end_slot=windowEndSlot)  # rho(t) for the poisson
        # now instead of sampling, consider that you will observe the mean, the mean of N(t) is rho(t)
        sampleProf[tp] = rho
    return sampleProf

def getEbarEmp(lamWin, listOS, window_length, windowFirstSlot, percentBooked):
    '''
    --------------------------------------------------------------------------------------------------------
    for a specific window (as determined by windowFirstSlot and lamWin) generate the standard dev.
    of instantaneous users at every time point, this corresponds to the standard dev. of the time-dependent
    R.V. representing the num. of users in an M/GI/infty queue that starts empty at the beg. of the window.
    The time-dependent R.V. has a time-dependent Poisson dist. that has a variance of rhot (mean=variance
    for Poisson dist), and so the standard dev. is sqrt(rhot)
    --------------------------------------------------------------------------------------------------------
    returns one standard deviation of the R.V. representing time-dependent instantaneous num. of users
    --------------------------------------------------------------------------------------------------------
    NOTE: THIS FUNCTION THINS LAMBDA BASED ON PERCENT BOOKED
    --------------------------------------------------------------------------------------------------------
    Note: computations are at time points 
    instead of time slots, t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    --------------------------------------------------------------------------------------------------------
    '''
    thinnedLambda = lamWin * (1 - percentBooked)  # this gives you the lambda for instantaneous
    windowEndSlot = windowFirstSlot + window_length - 1
    ebar = dict()  # store the standard deviation sampled profile
    for tp in np.arange(windowFirstSlot, windowEndSlot+1, 1):
        rho = getRhoTEmp(Lambda=thinnedLambda, listOS=listOS, timePt=tp, window_length=window_length, window_end_slot=windowEndSlot)  # standard dev. for the poisson
        # now instead of sampling, consider that you will observe the mean, the mean of N(t) is rho(t)
        ebar[tp] = np.sqrt(rho)
    return ebar



def computeTargetNEmp(lamWindow, listOS, window_end_slot, window_length, bookingProfile, percentBooked, tolerance, initialN=0):
    '''
    ------------------------------------------------------------------------------------------------------
    computes the target n for every window.
    ------------------------------------------------------------------------------------------------------
    Note that the booking profile should include instantaneous
    riders and book aheads that did not exist in the past time window
    bookingprofile=fBA+fp
    ------------------------------------------------------------------------------------------------------
    lamWindow is average arrival rate per slot over the window
    initialN is an initialization of the target n
    window_length is number of slots per window
    ------------------------------------------------------------------------------------------------------
    NOTE: THIS FUNCTION THINS THE ARRIVAL RATE BASED ON PERCENT BOOKED
    ------------------------------------------------------------------------------------------------------
    Note: computations are at time points 
    instead of time slots, t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    ------------------------------------------------------------------------------------------------------
    '''
    n = initialN
    satisfied = False  # boolean until we find n that satisfies requests
    while satisfied is False:
        window_complete = False
        probs = list()  # probability of exceeding in every time slot
        tp = window_end_slot - window_length + 1  # get the first time point in window t0| -- slot 1 -- |t1|, that would be t1
        while window_complete is False:
            probs.append(computeProbEmp(lamWindow*(1-percentBooked), listOS, tp, window_length, window_end_slot, bookingProfile, n))  # lamSlots and muSlots of first window
            tp += 1
            if tp == window_end_slot + 1:
                window_complete = True
        if np.mean(probs) < tolerance:
            satisfied = True
        else:
            n += 1
    return n



def computeTargetNTotalStep(lam, dictOSwin, dataDict, region, winFirstSlot, winEndSlot, window_length, percentBooked, tolerance, IRsoFar, BAsoFar, initialN=0):
    '''
    ------------------------------------------------------------------------------------------------------
    computes one step of computeTargetNTotalEmp (i.e., for one window only)
    generates the BA from data, maintains the InOut tuples, and computes the targets
    for a single window in a single region
    ------------------------------------------------------------------------------------------------------    
    input:
        IRsoFar: remaining IR from past windows
        BAsoFar: remaining BA from past windows
    output:
        target n
        IRInOut: IR that will be active and that will be realized during window
        BAInOut: BA that will be active during window
        BAprof: fBA+fp, a dictionary of sum of BA during window and IR from prev. windows that will be
                active during window, does not account for IR that will be realized during subsequent
                window
    ------------------------------------------------------------------------------------------------------
    Note: difference between BAprof and BAInOut; BAInOut is only for BA's while BAprof is fBA+fp which
    may include IR requests that inititated in prior windows and are now part of fp
    ------------------------------------------------------------------------------------------------------
    Note: computations are at time points 
    instead of time slots, t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    ------------------------------------------------------------------------------------------------------
    '''
    BAInOut = BAsoFar  # stores all the BA in and out prior to window of interest (remember we're looking at a specific region)
    IRInOut = IRsoFar  # stores the IR in and out prior to window of interest (remember we're looking at a specific region)
    entriesDictIR, entriesDictBA = retrieveEntries(dataDict, region, winFirstSlot, winEndSlot, percentBooked)  # get entries for the new window in region
    BAInOut.extend(InOutFromDict(entriesDictBA))  # add the BA's that will be observed in the upcoming window
    BAprofBAIO, BAInOut = generateBA(InOut=BAInOut, windowFirstSlot=winFirstSlot, windowLength=window_length)  # BA from past and upcoming BA's, also cleans BA to remove things that ended in the past
    BAprofIRIO, IRInOut = generateBA(InOut=IRInOut, windowFirstSlot=winFirstSlot, windowLength=window_length)  # BA from past IR, incoming IR do not go into this (still unknown), maintain IR from past windows to remove ones that have already exited before beginning of window
    BAprof = sumDicts(BAprofBAIO, BAprofIRIO)  # fp + fBA
    target_n = computeTargetNEmp(lamWindow=lam, listOS=dictOSwin, window_end_slot=winEndSlot, window_length=window_length, bookingProfile=BAprof, percentBooked=percentBooked, tolerance=tolerance, initialN=initialN)  # target n computaiton based on Poisson profile
    # now stochastics come in (real-life), and we add them to the IRInOut list, they may be used at upcoming windows as BA if they overflow to those windows
    IRInOut.extend(InOutFromDict(entriesDictIR))
    
    return target_n, IRInOut, BAInOut, BAprof



def computeTargetNTotalEmp(lamAcrossW, dictOS, dataDict, region, start_slot, end_slot, window_length, percentBooked, tolerance, initializationWindows=2, initialN=0):
    '''
    ------------------------------------------------------------------------------------------------------
    This implements target computation across all windows.
    
    WARNING: this method assumes all users are admitted, refer to 
    manage-demand-adm.py for admission control and target computations
    using computeTargetNTotalStep
    ------------------------------------------------------------------------------------------------------
    computes target windows across time windows for a specific region
    generates the BA from data, maintains the InOut tuples, and computes the targets
    for across window in a single region
    ------------------------------------------------------------------------------------------------------
    input:    
    lamAcrossW: MLE arrival rates for every time window
    dictOS: dictionary generated from getOrderedService that specifies for 
            every window and ordered list of service time (in units of slots)    
    dataDict: dictionary with the actual data, use this dictionary
            to generate the BA from actual data and to obtain the realizations
            of IR from actual data
    start_slot: first time slot to consider (including intial windows)
    end_slot: last time slot of last window
    initializationWindows: first couple of windows used to generate fp so that
            we have data from prior windows
    initialN: initialization for target N computations
    ------------------------------------------------------------------------------------------------------
    note that start_slot and end_slot must be multiples of the window_length
    ------------------------------------------------------------------------------------------------------
    returns: target n across windows, booking profile across windows (fp + fBA),
    realizations from data of observed IR, sum of IR (predicted) and BA (fp + fBA) across time windows
    ------------------------------------------------------------------------------------------------------
    the method also returns profiles that are needed for plotting
    sample profile: mean of the number of busy servers in an M/GI/infty queue, starts at zero at beg. 
                    of every window
    sample profile total: mean rhot shifted up by num. BA's (fBA+fp)
    error bars: np.sqrt(rhot)
    BAprof: for a given book-ahead level, the BA profile during a window (BA + fp)
    actual incoming profile IR: observed number of IR from data, considered as realization of M/GI/infty
    actual incoming profile total: IR shifted by BA (fBA+fp)
    ------------------------------------------------------------------------------------------------------
    Note: Instead of initialization windows, you can put in BAsoFar and IRsoFar as in 
          computeTargetNTotalStep
    ------------------------------------------------------------------------------------------------------
    Note: computations are at time points 
    instead of time slots, t1, t2, t3, t4
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    ------------------------------------------------------------------------------------------------------
    '''
    # store results
    sampleProf = dict()  # stores the mean of the R.V.s
    ebars = dict()  # stores one standard dev. of the R.V.s
    sampleProfTotal = dict()  # stores the mean of the R.V.s shifted up by fBA+fp
    BAprofStore = dict()  # stores fBA+fp
    actualIR = dict()  # stores the realization of IR
    actualProf = dict()  # stores the full profile from data, realization of IR and BA (fp+fBA)
    
    
    numWindows = (end_slot - start_slot + 1)/(window_length)  # total number of windows    
    
    BAInOut = list()  # stores all the BA in and out across time (note that this is only BAs generated or remaining in system)
    IRInOut = list()  # stores the IR in and out across time (note that these may become BA if they cross time windows but they are maintained separately)
    target_n = dict()  # stores target n values for windows of interest
    
    # go through data, look at initialization windows that have already passed, and determine users that already came in
    for win in np.arange(1, initializationWindows+1, 1):
        firstTimeSlot = start_slot + window_length*(win - 1)
        windowEndSlot = win * window_length
        prevEntriesDictIR, prevEntriesDictBA = retrieveEntries(dataDict, region, firstTimeSlot, windowEndSlot, percentBooked)  # retrieve dict with all customers that enter in time window
        IRInOut.extend(InOutFromDict(prevEntriesDictIR))  # note that it should not matter whether they get added to BA or IR since for windows of interest they will BA anyway
        BAInOut.extend(InOutFromDict(prevEntriesDictBA))
    
    # now for remaining windows compute target n
    for window in np.arange(initializationWindows+1, numWindows+1, 1):  # start from windows after the initialization windows
        windowEndSlot = window * window_length
        firstTimeSlot = start_slot + window_length*(window - 1)  # first time slot of the window

        # generate book-ahead for upcoming window and add it to BAInOut, use real data to generate that BA profile
        entriesDictIR, entriesDictBA = retrieveEntries(dataDict, region, firstTimeSlot, windowEndSlot, percentBooked)
        BAInOut.extend(InOutFromDict(entriesDictBA))  # add future BA to existing BA's InOut
        # NOTE: the book-aheads from previous windows that remain in the system are bundled with the book-aheads that initiate in the upcoming window, they're all in InOut!
        BAprofBAIO, BAInOut = generateBA(InOut=BAInOut, windowFirstSlot=firstTimeSlot, windowLength=window_length)  # get BA prof from BA's (BA portion of fp and fBA)
        BAprofIRIO, IRInOut = generateBA(InOut=IRInOut, windowFirstSlot=firstTimeSlot, windowLength=window_length)  # get BA prof from past IRs (IR portion of fp), NOTE WE DID NOT EXTEND IRS YET!!    
        BAprof = sumDicts(BAprofBAIO, BAprofIRIO)  # get total BA prof fBA+fp

        # compute target n
        target_n[window] = computeTargetNEmp(lamWindow=lamAcrossW[window], listOS=dictOS[window], window_end_slot=windowEndSlot, window_length=window_length, bookingProfile=BAprof, percentBooked=percentBooked, tolerance=tolerance, initialN=initialN)
        
        # now stochastics come in (real-life), and we add them to the IRInOut list
        IRInOut.extend(InOutFromDict(entriesDictIR))
        
        # update the profiles for plotting
        sampleProf.update(getSampleProfileEmp(lamWin=lamAcrossW[window], listOS=dictOS[window], window_length=window_length, windowFirstSlot=firstTimeSlot, percentBooked=percentBooked))
        ebars.update(getEbarEmp(lamWin=lamAcrossW[window], listOS=dictOS[window], window_length=window_length, windowFirstSlot=firstTimeSlot, percentBooked=percentBooked))
        BAprofStore.update(BAprof)
        sampleProfTotal.update(sumDicts(BAprof, getSampleProfileEmp(lamWin=lamAcrossW[window], listOS=dictOS[window], window_length=window_length, windowFirstSlot=firstTimeSlot, percentBooked=percentBooked)))
        actualObservedIR = generateBA(InOut=InOutFromDict(entriesDictIR), windowFirstSlot=firstTimeSlot, windowLength=window_length)[0]
        actualIR.update(actualObservedIR)
        actualProf.update(sumDicts(BAprof, actualObservedIR))
    
    return target_n, sampleProf, ebars, sampleProfTotal, BAprofStore, actualIR, actualProf




if __name__ == '__main__':
    # data initialization and analysis
    dDict, head = readCSV('data/ridesLyftMHTN14.csv')
    dDict = addTimeStamp(dDict, slotInMinutes=5)
    dictNum = computeNumberOfUsers(dDict)
    adjacency = {1:[2], 2:[1, 3, 4], 3:[2, 4], 4:[2, 3]}  # specifies from which regions you can move vehicles to adjacent regions, not needed in this method
    
    # get rates
    lamSlots1, lamMint1 = getLamPerRegionMLE(dataDict=dDict, Origin=1, slotInMinutes=5, windowInMinutes=20, numWindows=9, Destination=None)  # region 1    
    lamSlots2, lamMint2 = getLamPerRegionMLE(dataDict=dDict, Origin=2, slotInMinutes=5, windowInMinutes=20, numWindows=9, Destination=None)  # region 2    
    lamSlots3, lamMint3 = getLamPerRegionMLE(dataDict=dDict, Origin=3, slotInMinutes=5, windowInMinutes=20, numWindows=9, Destination=None)  # region 3    
    lamSlots4, lamMint4 = getLamPerRegionMLE(dataDict=dDict, Origin=4, slotInMinutes=5, windowInMinutes=20, numWindows=9, Destination=None)  # region 4
    
    # specify parameters
    pBook = 0.0
    tol = 0.01
    
    # compute target_n across all windows
    # WARNING: methods below assume all users are admitted, refer to manage-demand-adm.py for admission control and min cost flow dispatching
    # region 1
    dictOS1 = getOrderedService(dDict, region=1, slotInMinutes=5, windowInMinutes=20, numWindows=9)
    target_n1, sampleProf1, ebars1, sampleProfTotal1, BAprofStore1, actualIR1, actualProf1 = computeTargetNTotalEmp(lamSlots1, dictOS1, dDict, region=1, start_slot=1, end_slot=36, window_length=4, percentBooked=pBook, tolerance=tol, initializationWindows=2, initialN=0)
    # region 2
    dictOS2 = getOrderedService(dDict, region=2, slotInMinutes=5, windowInMinutes=20, numWindows=9)
    target_n2, sampleProf2, ebars2, sampleProfTotal2, BAprofStore2, actualIR2, actualProf2 = computeTargetNTotalEmp(lamSlots2, dictOS2, dDict, region=2, start_slot=1, end_slot=36, window_length=4, percentBooked=pBook, tolerance=tol, initializationWindows=2, initialN=0)
    # region 3
    dictOS3 = getOrderedService(dDict, region=3, slotInMinutes=5, windowInMinutes=20, numWindows=9)
    target_n3, sampleProf3, ebars3, sampleProfTotal3, BAprofStore3, actualIR3, actualProf3 = computeTargetNTotalEmp(lamSlots3, dictOS3, dDict, region=3, start_slot=1, end_slot=36, window_length=4, percentBooked=pBook, tolerance=tol, initializationWindows=2, initialN=0)
    # region 4
    dictOS4 = getOrderedService(dDict, region=4, slotInMinutes=5, windowInMinutes=20, numWindows=9)
    target_n4, sampleProf4, ebars4, sampleProfTotal4, BAprofStore4, actualIR4, actualProf4 = computeTargetNTotalEmp(lamSlots4, dictOS4, dDict, region=4, start_slot=1, end_slot=36, window_length=4, percentBooked=pBook, tolerance=tol, initializationWindows=2, initialN=0)
    
    
