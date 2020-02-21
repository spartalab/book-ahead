# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:07:17 2019

This code is for utilities such as managing data

@author: cesny
"""

import csv
import math
import numpy as np
import copy
from datetime import time


#-----------   data processing and visualization ---------------
def readCSV(file):
    '''
    read in csv file
    '''
    with open(file, mode='r') as infile:
        read = csv.reader(infile)
        head = next(read)
        dataDict = dict()
        for header in range(1, 12): dataDict[head[header]] = []  # create headers
        for row in read:
            for header in range(1, 12):
                dataDict[head[header]].append(row[header])
    return dataDict, head[1:]


def addTimeStamp(dataDict, slotInMinutes):
    '''
    ------------------------------------------------------------------------
    for each data point assign when it comes in
        based on conventinon that we operate by 
        timeSlots
    ------------------------------------------------------------------------
    data specific: starting hour is 16
        so first time slot is between 16:00:00 and 16:timeslot:00 = slot 1
    timeSlot numbering starts from 1
    ------------------------------------------------------------------------
    Note that the time slot corresponds to between two time points
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| --..
    ------------------------------------------------------------------------
    '''
    dataDict['TimeIn'] = list()
    for val in dataDict['Pickup_DateTime']:
        splitDate = val.split(' ')
        splitTimeIn = splitDate[1].split('-')[0].split(':')
        timeStampMinutes = (float(splitTimeIn[0]) - 16)*60 + float(splitTimeIn[1]) + float(splitTimeIn[2])*(1.0/60)  # time stamp starting from zero at 16:00:00
        timeSlot = math.ceil(timeStampMinutes/slotInMinutes)
        dataDict['TimeIn'].append(timeSlot)
    
    dataDict['TimeOut'] = list()
    for val in dataDict['DropOff_datetime']:
        splitDate = val.split(' ')
        splitTimeOut = splitDate[1].split('-')[0].split(':')
        timeStampMinutes = (float(splitTimeOut[0]) - 16)*60 + float(splitTimeOut[1]) + float(splitTimeOut[2])*(1.0/60)  # time stamp starting from zero at 16:00:00
        timeSlot = math.ceil(timeStampMinutes/slotInMinutes)
        dataDict['TimeOut'].append(timeSlot)
    return dataDict


def computeNumberOfUsers(dataDict, numRegions=4):
    '''
    ---------------------------------------------------------
    returns number of users active from every origin to every
    destination at the end of the time slot
    ---------------------------------------------------------
    Warning: addTimeStamp must be called before this function
    ---------------------------------------------------------
    Note that this returns the number of user at time points 
    instead of slots
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| --..
    dictofDicts[1] returns the number of users in the system
    at time t1
    thus, if users exit within slot 1 they are not counted 
    at dictofDicts[1]
    also, if a users enter in slot 1, and remains beyond slot 1
    they will be counted at t1
    ---------------------------------------------------------
    regions must be defined in input csv file, Manhattan is 
    divided into four regions numbered from 1-->4
    ---------------------------------------------------------
    '''
    # initialization
    dictofDicts = dict()
    for num in np.arange(1, numRegions+1, 1):
        dictofDicts[num] = dict()
        dictofDicts[num]['Total']= dict()
        for subnum in np.arange(1, numRegions+1, 1):
            dictofDicts[num][subnum] = dict()
        for timePt in list(np.arange(1,max(dataDict['TimeOut']),1)):
            for subnum in np.arange(1, numRegions+1, 1):
                dictofDicts[num][subnum][timePt] = 0
            dictofDicts[num]['Total'][timePt] = 0

    for key, timeIn in enumerate(dataDict['TimeIn']):
        timeOut = dataDict['TimeOut'][key]  # get the corresponding exit time
        origin = int(dataDict['region'][key])  # get corresponding origin
        destination = int(dataDict['DOregion'][key])  # get corresponding destination
        # add to customer to system count, don't include at last spot "timeOut" since they leave before it (time point perspective)
        for soujourn in np.arange(timeIn, timeOut, 1):  # soujourn is timePt
            dictofDicts[origin]['Total'][soujourn] += 1
            dictofDicts[origin][destination][soujourn] += 1

    return dictofDicts


def getNumTrips(dataDict, timeSlotIn1, timeSlotIn2, region=None):
    '''
    --------------------------------------------------------------------
    retrieves the number of trips that initiate between timeSlotIn1 and 
    timeSlotIn2
    --------------------------------------------------------------------
    if the region parameter is specified it only returns trips
    initiated in the region of interest
    --------------------------------------------------------------------
    timeSlotIn2 is included in the computation, i.e., trips that
    within timeSlotIn2 are counted
    --------------------------------------------------------------------
    '''
    Trips = 0
    if region is None:
        for timein in dataDict['TimeIn']:
            if timein in np.arange(timeSlotIn1, timeSlotIn2+1, 1):
                Trips+=1
    else:
        for key, timein in enumerate(dataDict['TimeIn']):
            if (timein in np.arange(timeSlotIn1, timeSlotIn2+1, 1)) and int(dataDict['region'][key]) == region:
                Trips+=1
    return Trips



def avgNumPerWindow(num, window_length):
    '''
    -----------------------------------------------
    takes number of vehicles in every slot
    finds average number per window
    num is a dictionary with {timePt: numUsers,..}
    -----------------------------------------------
    '''
    avgNum = dict()
    winCount=0
    store=list()
    for elem in num:
        if elem%window_length == 0:
            store.append(num[elem])
            winCount+=1
            avgNum[winCount] = np.mean(store)
            store=list()
        else:
            store.append(num[elem])
    return avgNum



def retrieveEntries(dataDict, region, firstTimeIn, lastTimeIn=None, percentBooked=None):
    '''
    ------------------------------------------------------------------------
    Given a dictionary,
        1- we return a dictionary with entries that start within slot 
        timeIn or start between slots timeIn and lastTimeIn inclusive
        2- if some of the data is booked returns two dictionaries with
        booked profile and IR, where you are assign to book or IR based on
        percentBook
    ------------------------------------------------------------------------
    Warning: addTimeStamp must be called before this function
    Be careful: returns 1 dict if percentBook is None, otherwise returns 2
    ------------------------------------------------------------------------
    This method is in units of slots!! (not time point based!)
    ------------------------------------------------------------------------
    '''
    if percentBooked is None:  # get total actual data in time window (do not sample it based on percentBooked)
        newDict = copy.deepcopy(dataDict)
        delIndexes = list()
        if lastTimeIn is None:  # get all the entries that arrive at a particular time slot
            for key, val in enumerate(dataDict['TimeIn']):
                if (val != firstTimeIn) or (int(dataDict['region'][key])!=region):
                    delIndexes.append(key)
        else:
            eligible_set = list(np.arange(firstTimeIn, lastTimeIn+1, 1))
            for key, val in enumerate(dataDict['TimeIn']):
                if (val not in eligible_set) or (int(dataDict['region'][key])!=region):
                    delIndexes.append(key)
        # clean dict
        for label in newDict:
            for index in sorted(delIndexes, reverse=True):
                del newDict[label][index]
        
        return newDict
    
    else:
        newDictIR = copy.deepcopy(dataDict)
        newDictBA = copy.deepcopy(dataDict)
        delIndexesIR = list()  # indexes to delete to get the IR dict (del. indexes that do not match time/region requirements and are BA)
        delIndexesBA = list()  # indexes to delete to get BA dict  (del. indexes that do not match time/region requirements and are BA)
        if lastTimeIn is None:  # get all the entries that start at a specific slot
            for key, val in enumerate(dataDict['TimeIn']):
                if (val != firstTimeIn) or (int(dataDict['region'][key])!=region):  # if value is not the slot of interest then delete it
                    delIndexesIR.append(key)
                    delIndexesBA.append(key)
                else:
                    rand = np.random.uniform(0,1)  # sample a uniform random variable
                    if rand>percentBooked:  # if you get an IR
                        delIndexesBA.append(key)  # delete the key from the BA dict
                    elif rand<=percentBooked:
                        delIndexesIR.append(key)
        else:
            eligible_set = list(np.arange(firstTimeIn, lastTimeIn+1, 1))
            for key, val in enumerate(dataDict['TimeIn']):
                if (val not in eligible_set) or (int(dataDict['region'][key])!=region):
                    delIndexesIR.append(key)  # delete if it is not in the eligible set or wrong origin
                    delIndexesBA.append(key)
                else:  # if it is in the eligible set and right origin
                    rand = np.random.uniform(0,1)  # sample a uniform random variable
                    if rand>percentBooked:  # if you get an IR
                        delIndexesBA.append(key)  # delete it from the BA dict
                    elif rand<=percentBooked:
                        # print(key)
                        delIndexesIR.append(key)
        # clean dictionaries
        for label in newDictIR:
            for index in sorted(delIndexesIR, reverse=True):
                del newDictIR[label][index]
        for label in newDictBA:
            for index in sorted(delIndexesBA, reverse=True):
                del newDictBA[label][index]

        return newDictIR, newDictBA


def generateBA(InOut, windowFirstSlot, windowLength):
    '''
    ----------------------------------------------------------------------------
    Gets the BA profile for a time window and an InOut profile.
    ----------------------------------------------------------------------------
    In reference to the paper, the BA profile is fp+fBA i.e., everything except
    instantaneous requests that will appear during the window.
    
    Thus, the InOut profile could correspond to BA's from current or past 
    windows or it could correspond to IRs remaining in the system from past
    windows
    ----------------------------------------------------------------------------    
    returns the BA profile as a dictionary over *time points*
    ----------------------------------------------------------------------------
    Note: if a customer is from a past window, then the windowFirstSlot will
    be the maximum below
    else: if he starts within the window (BA) then his entry time will be
    greater than or equal to the first slot
    ----------------------------------------------------------------------------
    Note that the edge time point (t4) is computed with the prev. window (win 1)
    since in the next window we only consider trips that start during or after
    the first time slot (slot 5) so we only consider time points 5-8 for trips
    that are continuing from before, even though they passed through time pt 4
    ----------------------------------------------------------------------------
    Note: that this returns the number of user at time points 
    instead of time slots
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| -- slot 3 --|t3| -- slot 4 --|t4|
    BAprof[1] returns the number of users in the system
    at time t1
    thus, if users exit within slot 1 they are not counted 
    at BAprof[1]
    elif a users enter in slot 1, and remains beyond slot 1
    they will be counted at t1
    e.g. users enters in slot 2 and exits in slot 4
        the user will be counted at t2 and t3 corresponding to
        BAprof[2] and BAprof[3]
    ----------------------------------------------------------------------------
    windowLength: number of slots per window
    windowFirstSlot: number of first slot in window
    InOut: list with (entry, exit, origin, destination) tuples
    ----------------------------------------------------------------------------
    '''
    BAprof = dict()
    windowEndSlot = windowFirstSlot + windowLength - 1  # window of interest
    for timePt in np.arange(windowFirstSlot, windowEndSlot+1, 1):  # initialize BA profile
        BAprof[int(timePt)] = 0
    # maintain IOs
    newInOut = maintainInOut(InOut, windowFirstSlot)
    
    for elem in newInOut:
        entryTimeSlot = elem[0] # entry slot of a specific customer, may have entered before beginning of window
        exitTimeSlot = elem[1]  # exit slot of a specific customer, must exit after beginning of window (maintainInOut)
        for soujourn in np.arange(max(windowFirstSlot, entryTimeSlot), min(windowEndSlot+1, exitTimeSlot), 1):  # time customer spends in system in CURRENT WINDOW (including end timePt)
            BAprof[int(soujourn)] += 1
    
    return BAprof, newInOut



def maintainInOut(currentIOprofile, windowFirstSlot):
    '''
    --------------------------------------------------------------------------
    This function is for maintanence. 
    The function removes entries that are expired. i.e., it removes
    (entry, exit, origin, destination) tuples where there exit slot is less
    than the first time point of the upcoming window
    --------------------------------------------------------------------------
    currentIOprofile is a list based on InOut profile
    windowFirstSlot is the starting slot of the new time window, and we care
        about users that will remain in the system within this new window
    --------------------------------------------------------------------------
    note that this function operates in units of slots
    --------------------------------------------------------------------------
    '''
    newIOprofile = copy.deepcopy(currentIOprofile)
    for elem in currentIOprofile:
        if elem[1] < windowFirstSlot:  # if the end slot is less than the beginning slot, remove the (entry, exit) element
            newIOprofile.remove(elem)
    return newIOprofile


def InOutFromDict(dataDict):
    '''
    -----------------------------------------------------
    gets InOut from dictionary
    -----------------------------------------------------
    InOut is in units of slots
    -----------------------------------------------------
    InOut stores timeIn, timeOut, InRegion, OutRegion
    -----------------------------------------------------
    '''
    # sanity
    if len(dataDict['TimeIn']) != len(dataDict['TimeOut']): print('... fatal error: InOutFromDict ...')
    
    InOut = list()
    for key, timeIn in enumerate(dataDict['TimeIn']):
        InOut.append((timeIn, dataDict['TimeOut'][key], int(dataDict['region'][key]), int(dataDict['DOregion'][key])))
    return InOut


def sumDicts(IRprofile, BAprofile):
    '''
    -----------------------------------------------------
    sums IR and BA profiles or BA and BA
    -----------------------------------------------------
    BA profile is a dictionary across time points!
    IR is a dictionary across time points as well
    -----------------------------------------------------
    Warning: make sure that both dictionaries have 
    appropriate keys and values
    -----------------------------------------------------
    units of time points
    -----------------------------------------------------
    '''
    sumProfile = dict()
    for elem in IRprofile:
        sumProfile[elem] = IRprofile[elem] + BAprofile[elem]
    return sumProfile


def diffDicts(ActualDict, ExpectedDict):
    '''
    gets the difference between expected profile and actual observed profile
    '''
    diffProfile = dict()
    for elem in ActualDict:
        diffProfile[elem] = ActualDict[elem] - ExpectedDict[elem]
    return diffProfile


def getPortionDict(dataDict, start_elem, end_elem):
    '''
    this function is for sanity checks
    retrieves elements of dict between start_elem and
    end_elem
    '''
    cutDict=dict()
    for elem in np.arange(start_elem, end_elem+1, 1):
        cutDict[elem] = dataDict[elem]
    return cutDict


def map2time(timePt, slotInMinutes=5, startTime=time(hour=16, minute=00, second=00)):
    """
    ------------------------------------------------
    maps time point to clock time
    ------------------------------------------------
    startTime: time at which data is collected
    slotInMinutes: length of slot in minutes
    timePt: time point
    ------------------------------------------------
    """
    # first check if we need to add any hours
    addHour = math.floor(timePt*slotInMinutes/60)
    # check if we need to add any minutes on top of that
    addMinute = timePt*slotInMinutes - addHour*60
    # add to time
    newHour = startTime.hour + addHour
    newMinute = startTime.minute + addMinute
    newTime = time(hour=newHour, minute=newMinute, second=00)
    return newTime

#-----------------------------------------------------------------------------


if __name__ == '__main__':
    print('utils')