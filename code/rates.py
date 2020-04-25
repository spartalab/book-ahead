# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:38:32 2019

This module is used for defining functions that compute arrival and 
    service rates
    
@author: cesny
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.stats as st
from utils import readCSV, addTimeStamp, computeNumberOfUsers, getNumTrips, avgNumPerWindow, retrieveEntries, generateBA, maintainInOut, InOutFromDict, sumDicts, diffDicts, getPortionDict, map2time



#-- calibration of parameters and generation of booking profiles -------------

def getOrderedService(dataDict, region, slotInMinutes, windowInMinutes, numWindows):  # has to be by origin!!
    '''
    --------------------------------------------------------------------------------------------------------
    returns the list of service times ordered from smallest to largest
    this will be used to generate the empirical service distribution
    --------------------------------------------------------------------------------------------------------
    returns a dictionary with ordered list for each window, dictofLists[window num.]=[least ser. time, ...]
    --------------------------------------------------------------------------------------------------------
    '''
    dictofLists = dict()  # dictionary stores for every window soujourn time
    for window in np.arange(1, numWindows+1, 1):
        dictofLists[window] = list()
    for key, In in enumerate(dataDict['Pickup_DateTime']):
        if int(dataDict['region'][key]) == region:  # make sure you're operating in the right region!
            splitDateIn = In.split(' ')
            splitTimeIn = splitDateIn[1].split('-')[0].split(':')  # get the time you enter
            splitDateOut = dataDict['DropOff_datetime'][key].split(' ')
            splitTimeOut = splitDateOut[1].split('-')[0].split(':')  # get the time you leave
            hourDiff = float(splitTimeOut[0]) - float(splitTimeIn[0])
            minuteDiff = float(splitTimeOut[1]) - float(splitTimeIn[1])
            secondDiff = float(splitTimeOut[2]) - float(splitTimeIn[2])
            totalMinuteDiff = hourDiff*60 + minuteDiff + secondDiff*(1.0/60)  # total difference in minutes between entry and exit time
            totalMinuteDiffSlots = totalMinuteDiff / slotInMinutes  # get the service time in slots
            
            timeStampIn = (float(splitTimeIn[0]) - 16)*60 + float(splitTimeIn[1]) + float(splitTimeIn[2])*(1.0/60)  # time you enter system in minutes
            timeStampLocation = math.ceil(timeStampIn/windowInMinutes)   # determine which window you belong to
            dictofLists[timeStampLocation].append(totalMinuteDiffSlots)  # add the difference in entry and exit to dictofLists based on entry location
    for elem in dictofLists:
        dictofLists[elem].sort()
    return dictofLists


def checkPoisson(dataDict, Origin, slotInMinutes, windowInMinutes, numWindows, window, Destination=None):
    '''
    ------------------------------------------------------------------------------------------------------
    1. generates a probabilities for the poisson distribution using the MLE parameter
    2. compares with frequency distribution derived from the data
    ------------------------------------------------------------------------------------------------------
    '''
    lamSlots, lamMint = getLamPerRegionMLE(dataDict, Origin, slotInMinutes, windowInMinutes, numWindows, Destination=None)
    lamInterest = lamMint[window]
    print('Poisson rate lambda: ', lamInterest)
    # now get data, number of arrivals per minute across minutes for time window of interest
    firstMinute = windowInMinutes*(window-1)+1
    lastMinute = windowInMinutes*window
    countDict = dict()  # initialize count dict, stores for every minute num. of arrivals
    for ind in np.arange(firstMinute, lastMinute+1, 1):
        countDict[ind]=0
    for key, In in enumerate(dataDict['Pickup_DateTime']):
        if int(dataDict['region'][key]) == Origin:  # only count the number of arrivals for region on interest
            splitDateIn = In.split(' ')
            splitTimeIn = splitDateIn[1].split('-')[0].split(':')  # get the time you enter
            timeStampIn = (float(splitTimeIn[0]) - 16)*60 + float(splitTimeIn[1]) + float(splitTimeIn[2])*(1.0/60)  # time you enter system in minutes
            if math.ceil(timeStampIn) in np.arange(firstMinute, lastMinute+1, 1):
                countDict[math.ceil(timeStampIn)]+=1
    print('countDict: ', countDict)
    print(' ')
    setofNumArrivals = set()  # stores all unique values that you observe of number of arrivals per minute
    tally = dict()
    for elem in countDict:
        tally.setdefault(countDict[elem], 0)  # sets to zero if not initialized
        tally[countDict[elem]] += 1/(windowInMinutes)
        setofNumArrivals.add(countDict[elem])
    print('tally: ', tally)
    # max likelihood for data
    maxlike = 0
    for elem in countDict:
        maxlike += countDict[elem]
    maxlike = maxlike/len(countDict)
    print('maximum like: ', maxlike)
    print('lambda used: ', lamInterest)
    # get expected probabilities
    probs = dict()
    for index in np.arange(min(setofNumArrivals), max(setofNumArrivals)+1, 1):
        probs[index]=st.poisson.pmf(index, lamInterest)
    return tally, probs


def getMu(dataDict, slotInMinutes, windowInMinutes, numWindows):
    '''
    ------------------------------------------------------------------------------------------------------
    returns the maximum likelihood estimator of the mean service rate assuming exponential distribution
    ------------------------------------------------------------------------------------------------------
    NOTE: IN REST OF CODE WE USE THE EMPIRICAL SERVICE TIME DISTRIBUTION, THE EXPONENTIAL DIST. TURNS 
    OUT TO BE INADEQUATE! (if an exp. dist. is used, the predicted num. of active users
    underestimates observed realization of num. of active users)
    ------------------------------------------------------------------------------------------------------
    for all vehicles that originate in a time window, compute their average service time from the data
    and derive mu as 1/time
    ------------------------------------------------------------------------------------------------------
    returns the mean service time in units of slots (muSlotsDict) and also in minutes (muDict)
    ------------------------------------------------------------------------------------------------------
    '''
    dictofLists = dict()  # dictionary stores for every window a list of soujourn time
    for window in np.arange(1, numWindows+1,1):
        dictofLists[window] = list()
        
    for key, val in enumerate(dataDict['Pickup_DateTime']):
        splitDateIn = val.split(' ')
        splitTimeIn = splitDateIn[1].split('-')[0].split(':')  # get the time you enter
        splitDateOut = dataDict['DropOff_datetime'][key].split(' ')
        splitTimeOut = splitDateOut[1].split('-')[0].split(':')  # get the time you leave
        hourDiff = float(splitTimeOut[0]) - float(splitTimeIn[0])
        minuteDiff = float(splitTimeOut[1]) - float(splitTimeIn[1])
        secondDiff = float(splitTimeOut[2]) - float(splitTimeIn[2])
        totalMinuteDiff = hourDiff*60 + minuteDiff + secondDiff*(1.0/60)  # total difference in minutes between entry and exit time
        
        timeStampIn = (float(splitTimeIn[0]) - 16)*60 + float(splitTimeIn[1]) + float(splitTimeIn[2])*(1.0/60)  # time you enter system in minutes
        timeStampLocation = math.ceil(timeStampIn/windowInMinutes)   # determine which window you belong to
        dictofLists[timeStampLocation].append(totalMinuteDiff)  # add the difference in entry and exit to dictofLists based on entry location
    
    meanTimeDict = dict()  # dictionary storing mean service time in minutes for every time window
    muDict = dict()  # dictionary sotring mu in minutes for every time window
    meanSlotsDict = dict()  # dictionary storing mean service time in unit of slots
    muSlotsDict = dict()  # dictionary storing service rate in unit of slots
    for window in np.arange(1, numWindows+1, 1):
        meanTimeDict[window] = np.mean(dictofLists[window])
        muDict[window] = 1.0/meanTimeDict[window]
        meanSlotsDict[window] = meanTimeDict[window]/slotInMinutes
        muSlotsDict[window] = 1.0/meanSlotsDict[window]
        
    return muSlotsDict, muDict


def getLambdaMLE(dataDict, slotInMinutes, windowInMinutes, numWindows):
    '''
    ------------------------------------------------------------------------------------------------------
    returns the maximum likelihood estimator of the Poisson arrival rate from the data
    ------------------------------------------------------------------------------------------------------
    returns 
    LambdaSlots: MLE arrival rate for each window in units of slots
    Lambda: MLE arrival rate for each window in units of minutes
    ------------------------------------------------------------------------------------------------------
    '''
    ArrivalsDict = dict()  # dictionary stores for every window total number of arrivals
    for window in np.arange(1, numWindows+1,1):  # initialize number of arrivals per window
        ArrivalsDict[window] = 0
    # tally how many you see per minute
    for val in dataDict['Pickup_DateTime']:
        splitDateIn = val.split(' ')
        splitTimeIn = splitDateIn[1].split('-')[0].split(':')
        timeStampMinutes = (float(splitTimeIn[0]) - 16)*60 + float(splitTimeIn[1]) + float(splitTimeIn[2])*(1.0/60)  # time stamp starting from zero at 16:00:00
        timeStampWindowLocation = math.ceil(timeStampMinutes/windowInMinutes)  # figure out which window arrival is in
        ArrivalsDict[timeStampWindowLocation] += 1  # add one arrival to that window
    Lambda = dict()  # stores lambda in units of arrivals/minute
    LambdaSlots = dict()  # stores lambda in units of arrivals/slot
    numSlots = windowInMinutes/slotInMinutes
    for window in np.arange(1, numWindows+1, 1):
        Lambda[window] = ArrivalsDict[window]/windowInMinutes
        LambdaSlots[window] = ArrivalsDict[window]/numSlots
        
    return LambdaSlots, Lambda
    
    
def getMuPerRegion(dataDict, Origin, slotInMinutes, windowInMinutes, numWindows, Destination=None):
    '''
    ------------------------------------------------------------------------------------------------------
    computes MLE estimator for service rate for a specific region-region pair
    ------------------------------------------------------------------------------------------------------
    e.g. mu12, mu11, mu21, mu22
    Origin Destination specify the rate we want
    if Destination is None, compute mu1, rate for total vehicles out of region1
    ------------------------------------------------------------------------------------------------------
    '''
    newDict = copy.deepcopy(dataDict)
    indexes = list()
    # get the indexes to be removed from the dictionary
    if Destination is None:
        for key, val in enumerate(dataDict['region']):
            if int(val) != Origin:  # not the Origin we are looking for
                indexes.append(key)
    else:
        for key, val in enumerate(dataDict['region']):
            if (int(val) != Origin) or (int(dataDict['DOregion'][key]) != Destination):  # not the specific OD pair we are looking for
                indexes.append(key)
    # now go through all lists in newDict and remove elements that don't conform
    for label in newDict:
        for index in sorted(indexes, reverse=True):
            del newDict[label][index]
    # now newDict has only the origin or OD pair we are interested in
    muSlots, mu = getMu(newDict, slotInMinutes, windowInMinutes, numWindows)
    return muSlots, mu


def getLamPerRegionMLE(dataDict, Origin, slotInMinutes, windowInMinutes, numWindows, Destination=None):
    '''
    ------------------------------------------------------------------------------------------------------
    computes MLE estimator for arrival rate lambda for a specific region-region pair
    if the Destination is set to None, then this methods computes the arrival rate MLE for a Poisson
    dist. where the requests initiate in the specified Origin region
    ------------------------------------------------------------------------------------------------------
    e.g. lam12, lam11, lam21, lam22
    Origin Destination specify the rate we want
    if Destination is None, compute lam1, rate for total vehicles that initiate in region 1
    ------------------------------------------------------------------------------------------------------
    '''
    newDict = copy.deepcopy(dataDict)
    indexes = list()
    # get the indexes to be removed from the dictionary
    if Destination is None:
        for key, val in enumerate(dataDict['region']):
            if int(val) != Origin:  # not the Origin we are looking for
                indexes.append(key)
    else:
        for key, val in enumerate(dataDict['region']):
            if (int(val) != Origin) or (int(dataDict['DOregion'][key]) != Destination):  # not the specific OD pair we are looking for
                indexes.append(key)
    # now go through all lists in newDict and remove elements that don't conform
    for label in newDict:
        for index in sorted(indexes, reverse=True):
            del newDict[label][index]
    # now newDict has only the origin or OD pair we are interested in
    lamSlots, Lambda = getLambdaMLE(newDict, slotInMinutes, windowInMinutes, numWindows)
    return lamSlots, Lambda


def getArrivalRates(infile1, infile2, infile3, Origin, slotInMinutes=5, windowInMinutes=20, numWindows=9, Destination=None):
    '''
    ------------------------------------------------------------------------------------------------------
    returns the arrival rates using data from past days
    infile1, infile2, and infile3 are csv files containing trip info. on similar days
    For NYC data: Feb 07, Feb 14, and Feb21 same hour intervals
    ------------------------------------------------------------------------------------------------------
    '''
    dDict1, head1 = readCSV(infile1)  # read data on previous days
    dDict2, head2 = readCSV(infile2)
    dDict3, head3 = readCSV(infile3)
    lamSlots1, lamMint1 = getLamPerRegionMLE(dDict1, Origin, slotInMinutes, windowInMinutes, numWindows, Destination)
    lamSlots2, lamMint2 = getLamPerRegionMLE(dDict2, Origin, slotInMinutes, windowInMinutes, numWindows, Destination)
    lamSlots3, lamMint3 = getLamPerRegionMLE(dDict3, Origin, slotInMinutes, windowInMinutes, numWindows, Destination)
    lamSlots = dict()
    lamMint = dict()
    for elem in lamSlots1:
        lamSlots[elem] = np.mean([lamSlots1[elem], lamSlots2[elem], lamSlots3[elem]])
        lamMint[elem] = np.mean([lamMint1[elem], lamMint2[elem], lamMint3[elem]])
    return lamSlots, lamMint, lamSlots1, lamSlots2, lamSlots3



def getArrivalRatesSingle(infile1, Origin, slotInMinutes=5, windowInMinutes=20, numWindows=9, Destination=None):
    '''
    ------------------------------------------------------------------------------------------------------
    gets the arrival rates using 
    data from a single past day observations
    NYC data: use Feb07, 2018 data to predict Feb14, 2018 rates
    ------------------------------------------------------------------------------------------------------
    '''
    dDictArate, headArate = readCSV(infile1)
    lamSlots, lamMint = getLamPerRegionMLE(dDictArate, Origin, slotInMinutes, windowInMinutes, numWindows, Destination)
    return lamSlots, lamMint


def getOrderedServiceSingle(infile1, region, slotInMinutes=5, windowInMinutes=20, numWindows=9):
    '''
    ------------------------------------------------------------------------------------------------------
    gets the service distribution from past data
    ------------------------------------------------------------------------------------------------------
    '''
    dDictSer, headSer = readCSV(infile1)
    dictofLists = getOrderedService(dDictSer, region, slotInMinutes, windowInMinutes, numWindows)
    return dictofLists


#-----------------------------------------------------------------------------    


if __name__ == '__main__':
    print('rates module')


