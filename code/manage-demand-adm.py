# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 20:52:14 2019

methods for managing supply
implements proposed framework
1- target predictin
2- admission control
3- min-cost flow

@author: cesny
"""
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import scipy.stats as st
from utils import readCSV, addTimeStamp, computeNumberOfUsers, getNumTrips, avgNumPerWindow, retrieveEntries, generateBA, maintainInOut, InOutFromDict, sumDicts, diffDicts, getPortionDict, map2time
from rates import getOrderedService, checkPoisson, getMu, getLambdaMLE, getMuPerRegion, getLamPerRegionMLE, getArrivalRates, getArrivalRatesSingle, getOrderedServiceSingle
from targetn import getEmpiricalIntegral, getRhoTEmp, computeCDFEmp, computeProbEmp, getSampleProfileEmp, getEbarEmp, computeTargetNEmp, computeTargetNTotalStep, computeTargetNTotalEmp
from ortools.graph import pywrapgraph



def manageSupply(lamWR, dictOSWR, dataDict, start_slot, end_slot, window_length, percentBooked, tolerance, initialSupplyDict, adjacency, tolDef=0, initialN=0):
    '''
    ------------------------------------------------------------------------------------------------------
    manages supply: 1. computes target 2. admission control 3. min. cost flow rebalancing
    ------------------------------------------------------------------------------------------------------
    input:
    lamWR: lambda for each region each window lamWR[region][window]
    dictOSWR: dictOS for each region each window (ordered service time) dictOSWR[region][window]
    dataDict: dictionary with data -- check utils.py
    start_slot: first slot for consideration
    end_slot: last slot for consideration
    window_length: length of one time window
    percentBooked: percent of requests that are BA
    tolerance: tolerance (QoS) level delta in computing target n
    initialSupplyDict: a dictionary giving the initial distribution of supply (can obtain from data) 
                      {region1:{'active': 49, 'idle': 2, 'arrived':0, 'departed': 0},...
                       note that at the beginning of a window arrived and departed are always equal to
                       zero since you have just reset the counts by bringing in external supply
    adjacency: adjacency dictionary giving which regions are adjacent to other
    tolDef: tolerable deficit L for load-balancing (by default considered zero as in paper)
    ------------------------------------------------------------------------------------------------------
    Note: difference between BAprof and BAInOut; BAInOut is only for BA's while BAprof is fBA+fp which
    may include IR requests that inititated in prior windows and are now part of fp
    ------------------------------------------------------------------------------------------------------
    '''
    # store simulation data
    sampleProf = dict()   # stores the instantaneous requests sample profile (mean of the N(t) process)
    ebars = dict()  # stores error bars on the sample profile path (sqrt mean)
    sampleProfTotal = dict()  # add the BA (fP+fBA) to the mean of the N(t) process
    BAprofStore = dict()  # stores the BA profile (fp+fBA)
    actualIR = dict()  # stores realization (from data) of the num. of users in M/GI/infty (removing randomly a certain percentage corresponding to BA)
    actualProf = dict()  # stores what was observed from the data (N(t)+fBA+fp)
    BAprofPortion = dict()  # stores the BA profile (instantaneous from previous windows and book-aheads fp+fBA) at the current iteration
    BAInOut = dict()  # stores all the BA in and out across time, note that it's a dict since considers diff. regions
    IRInOut = dict()  # stores the IR in and out across time, note that it's a dict since considers diff. regions
    target_n = dict()  # stores target n values for windows of interest
    deltas = dict()  # stores the deviation from target across time points at which load-balancing is done {timePt:{region1: 2, region2: 4, ..}}
    supply=dict()  # dict of dicts, supply = {timePt:{region:{active:2, idle:3, arrived:0, departed:0}}}
    supply[0] = initialSupplyDict  # supply initially at timePt zero
    blocked=dict()
    
    # initialize dicts
    for region in adjacency:
        sampleProf[region] = dict()
        ebars[region] = dict()
        sampleProfTotal[region] = dict()
        BAprofStore[region] = dict()
        actualIR[region] = dict()
        actualProf[region] = dict()
        BAInOut[region] = list()
        IRInOut[region] = list()
        target_n[region] = dict()
           
    numWindows = (end_slot - start_slot + 1)/(window_length)  # total number of windows
    
    # simulations across windows
    for window in np.arange(1, numWindows+1, 1):
        windowEndSlot = window * window_length  # last time slot of the window
        firstTimeSlot = start_slot + window_length*(window - 1)  # first time slot of the window
        # compute target n for every region at window of interest
        print('window', window)
        for region in adjacency:
            target_n[region][window], IRInOut[region], BAInOut[region], BAprof = computeTargetNTotalStep(lamWR[region][window], dictOSWR[region][window], dataDict, region, 
                    firstTimeSlot, windowEndSlot, window_length, percentBooked, tolerance, IRInOut[region], BAInOut[region], initialN=0)  # Note that IRInOut had been extended to include future entries that will appear throughout the window!
            # update the profiles needed for plotting
            print('BA profile region', region)
            print(BAprof)
            sampleProf[region].update(getSampleProfileEmp(lamWin=lamWR[region][window], listOS=dictOSWR[region][window], window_length=window_length, windowFirstSlot=firstTimeSlot, percentBooked=percentBooked))
            ebars[region].update(getEbarEmp(lamWin=lamWR[region][window], listOS=dictOSWR[region][window], window_length=window_length, windowFirstSlot=firstTimeSlot, percentBooked=percentBooked))
            BAprofStore[region].update(BAprof)  # stores BA profile for book-ahead and the previous i.e. fBA + fp
            sampleProfTotal[region].update(sumDicts(BAprof, getSampleProfileEmp(lamWin=lamWR[region][window], listOS=dictOSWR[region][window], window_length=window_length, windowFirstSlot=firstTimeSlot, percentBooked=percentBooked)))
            BAprofPortion[region] = BAprof  # stores BA profile for book-ahead and the previous i.e. fBA + fp, every window you re-write this, used to maintain current BAprof across regions
            
        print('BA prof store')
        print(BAprofStore)
        supply, delta = updateSupply(supply, adjacency, window, firstTimeSlot-1, target_n, tolDef, adjustTotal=True)  # note that the timePt is firstTimeSlot-1 since before window starts!
        deltas.update(delta)
        # now go through the window while maintaining supply throughout (accounting for exits and incoming) and computing blocked requests, we also clean the InOut at every timePt to remove those that were not admitted, this enables better tracking of arrivals/departures and available drivers across regions, and the BA profile at the subsequent window would only include those that were admitted in current window and remaining thereafter (in addition to BA's that will initiate in the future)
        for timePt in np.arange(firstTimeSlot, windowEndSlot+1, 1):
            blocked[timePt]=dict()
            supply[timePt]=dict()
            for region in adjacency:  # go through regions and do admission control/update for each region, as long as ride length > slot length then it doesn't matter order in which region supply is updated at a timePt since incoming rides from external regions would have initiated at prior timePts and so they would have been already blocked/admitted and so we would know incoming idle at every region and it is independent of what is CONCURRENTLY happening in other regions since they can't become available in region within the current timePt
                supply[timePt][region]=dict()
                BAInOutExternal=list()
                IRInOutExternal=list()
                for otherRegion in adjacency:
                    if otherRegion != region:
                        BAInOutExternal.extend(BAInOut[otherRegion])
                        IRInOutExternal.extend(IRInOut[otherRegion])
                if timePt == firstTimeSlot:
                    supply, blocked[timePt][region], IRInOut[region], BAInOut[region] = maintainSupply(supplyDict=supply, region=region, currentPt=timePt, winFirstSlot=firstTimeSlot, winLen=window_length, winEnd=windowEndSlot, IRInOut=IRInOut[region], BAInOut=BAInOut[region], IRInOutExternal=IRInOutExternal, BAInOutExternal=BAInOutExternal, BAprofile=BAprofPortion[region], target=target_n[region][window], windowFirstSlotFlag=True)  # note that maintain supply is implemented per region, as opposed to update supply which considers all regions simultaneously via min cost flow opt. program
                else:
                    supply, blocked[timePt][region], IRInOut[region], BAInOut[region] = maintainSupply(supplyDict=supply, region=region, currentPt=timePt, winFirstSlot=firstTimeSlot, winLen=window_length, winEnd=windowEndSlot, IRInOut=IRInOut[region], BAInOut=BAInOut[region], IRInOutExternal=IRInOutExternal, BAInOutExternal=BAInOutExternal, BAprofile=BAprofPortion[region], target=target_n[region][window], windowFirstSlotFlag=False)
            if timePt == (firstTimeSlot+1):
                supply, delta = updateSupply(supply, adjacency, window, timePt, target_n, tolDef, adjustTotal=False)
        # At any specific timePt above, the IRInOut and BAInOut are ajusted to account for admission up to the previous timePt, this implies that entries that initiate after timePt-1 will be in the InOut list (since it was extended in computeTargetNTotalStep) but we do not know if they will be admitted or not, we will know if they are admitted or not when we arrive to the timePt after the slot in which they initiate, and that is based on the available supply and previously admitted users
        IRInOutAdmitted = copy.deepcopy(IRInOut)  # retrieve all the entries that were admitted in the CURRENT window, recall that IRInOut could still include elements that started in earlier windows, for plotting purposes, we are only interested in ones that were initiated and admitted in current window, the IRInOut will then be trimmed again once more when we computeTargetNTotalStep at the subsequent window
        for region in adjacency:
            for InOutElem in IRInOut[region]:
                if InOutElem[0] not in np.arange(firstTimeSlot, windowEndSlot+1):  # not admitted before
                    IRInOutAdmitted[region].remove(InOutElem)  # we are trying to get what was admitted during the window, i.e., removing those admitted in prior windows but whose duration extends to the current window
            actualIR[region].update(generateBA(IRInOutAdmitted[region], firstTimeSlot, window_length)[0])  # add the admitted ones ot the IR
            actualProf[region].update(sumDicts(BAprofPortion[region], generateBA(IRInOutAdmitted[region], firstTimeSlot, window_length)[0]))
        
    return supply, blocked, target_n, deltas, sampleProf, ebars, BAprofStore, sampleProfTotal, actualIR, actualProf, IRInOut, BAInOut




def maintainSupply(supplyDict, region, currentPt, winFirstSlot, winLen, winEnd, IRInOut, BAInOut, IRInOutExternal, BAInOutExternal, BAprofile, target, windowFirstSlotFlag=False):
    '''
    ------------------------------------------------------------------------------------------------------
    maintains supply: propagates supplyDict forward by taking into consideration
    arrivals and departures, also implements admission control and computes blocked users 
	that are not able to find a driver
    
    the main purpose of this method is twofold:
        1- update idle and active in each region based on arrivals and dep. and this is essential for
        the update stage where you compute the deviation of (idle+active) from the target.
        2- adm. control and determine blocked passengers + update InOut lists to eliminate blocked rides
    ------------------------------------------------------------------------------------------------------
    Note: this function is implemented at every time step, you must implement
    it between load-balancing update points, a point that is about to be updated
    must have been maintained before so that we know how many vehicles are available
    to be moved around or how many vehicles we need to bring in
    ------------------------------------------------------------------------------------------------------
    Remember: IRInOut and BAInOut are by separated by region such that all the IR that initiate in
    a single region r are in IRInOut[r] and all the BA that originate in a single region rr are in 
    BAInOut[rr]
    IRInOut: IR that originated in the region
    BAInOut: BA that originated in the region
    IRInOutExternal: IR that originated outside the region during the window
    BAInOutExternal: BA that originated outside the region during the window
    InOut tuples are made of: 
    (dataDict['TimeIn'], dataDict['TimeOut'][key], int(dataDict['region'][key]), 
         int(dataDict['DOregion'][key]))
    ------------------------------------------------------------------------------------------------------
    NOTE: a critical assumption of associated with the dicretization simulation procedure is that 
    the duration of any ride should be longer than the slot duration 
    (reasonable assumption similar to the Courant–Friedrichs-Lewy (CFL) condition for
    the cell transmission model). The slot duration is 5 minutes, so we expect each ride to be longer
    than 5 minutes.
    ------------------------------------------------------------------------------------------------------
    Note: when this function is called, the InOut would have been already maintained at prior timePts,
    this means that for each region, we would have overwritten InOut, and we would only keep the 
    elements that exit within or beyond the time window of interest. In addition, the rides that were 
    blocked at earlier timePts would be removed from the InOut lists (across all regions).
    ------------------------------------------------------------------------------------------------------
    Note: we assume that vehicles that become free within a slot in a region will be available to 
    serve users whose requests initiate within the same slot and region. The slot duration is 5 minutes,
    so vehicles that come within the 5 minutes will be able to serve users that originate within the 5 
    minutes
    ------------------------------------------------------------------------------------------------------
    '''
    newSupplyDict = copy.deepcopy(supplyDict)
    if windowFirstSlotFlag is True:  # reset the arrived and departed, note that they're just used for book-keeping anyway
        newSupplyDict[currentPt][region]['arrived'] = 0 
        newSupplyDict[currentPt][region]['departed'] = 0
        newSupplyDict[currentPt][region]['idle'] = newSupplyDict[currentPt-1][region]['idle']  # initialize as being whatever happened at the previous point, this will be adjusted below based on what currently occurred
        newSupplyDict[currentPt][region]['active'] = newSupplyDict[currentPt-1][region]['active']
    else:
        newSupplyDict[currentPt][region]['arrived'] = newSupplyDict[currentPt-1][region]['arrived']
        newSupplyDict[currentPt][region]['departed'] = newSupplyDict[currentPt-1][region]['departed']
        newSupplyDict[currentPt][region]['idle'] = newSupplyDict[currentPt-1][region]['idle']
        newSupplyDict[currentPt][region]['active'] = newSupplyDict[currentPt-1][region]['active']
        
    # create the forwardBA = fp+fBA+fA
    IRInOutAdmitted=copy.deepcopy(IRInOut)  # store the ones that were already admitted by time currentPt-1, i.e. admitted since beginning of window up to current point, the ones that were admitted in previous windows are in BAprof, the ones that were blocked in prev. slots were removed from IRInOut at an earlier timePt
    for InOutElem in IRInOut:
        if InOutElem[0] not in np.arange(winFirstSlot, currentPt):  # not admitted before
            IRInOutAdmitted.remove(InOutElem)
    BAfromAdmittedIR = generateBA(IRInOutAdmitted, winFirstSlot, winLen)[0]  # now generate the BA associated with that for the current window
    forwardBA = sumDicts(BAfromAdmittedIR, BAprofile)  # has BA in window, BA from admitted IR in window, BA from admitted IR or BA in past windows, operates at time points
    
    # initialize
    arrived=0
    departed=0
    blocked = 0  # stores how many requests will be blocked in this region
    
    # considers the arrivals from other regions, those are only used for arrivals to see if you have gained idle drivers
    aggExtInOut = copy.deepcopy(IRInOutExternal)  # Note that here we implicitly consider only the admitted external rides, since blocked rides would have been removed from IRInOut and BAInOut at an earlier stage and they wouldn't show up here
    aggExtInOut.extend(BAInOutExternal)  # note that for InOut from other regions I don't care about what time they originate or from where, all I care is that they're ending up at the current slot in region
    for element in aggExtInOut:  # this is going through InOut from other regions to see if we have any arrivals
        if element[1] == currentPt:  # this means that the entry had exited at the current time point
            if element[3]  == region:  # this means that it had exited in the region of interest
                newSupplyDict[currentPt][region]['idle'] += 1  # when counting the number of vehicles in the region make sure you do not overcount this
                newSupplyDict[currentPt][region]['arrived'] += 1  # note that for now we only care about the region we're interested in, other regions will be revisited
                arrived+=1
        
    #  bring in the IRInOut from within region that finish at the current timePt so that you can use them as idle, note that if they finish at the current timePt it means they have been admitted earlier
    for element in IRInOut:
        if element[1] == currentPt:  # this means that the vehicles just exit (it was active before)
            if element[3] == region:  # this means they stayed in the region
                newSupplyDict[currentPt][region]['idle'] += 1  # not double counting since you still have the vehicle you just changed its state
                newSupplyDict[currentPt][region]['active'] -= 1
            elif element[3] != region:
                newSupplyDict[currentPt][region]['active'] -= 1  # you left the region, remove from active and do not add back to idle, it's lost, will be considered as incoming idle at other region
                newSupplyDict[currentPt][region]['departed'] += 1
                departed+=1
                
    #  bring in the BAInOut from within region that finish at the current timePt so that you can use them as idle, note that if they finish at the current timePt it means they have been admitted earlier
    for element in BAInOut:
        if element[1] == currentPt:  # this means that the vehicles just exit (it was active before)
            if element[3] == region:  # this means they stayed in the region
                newSupplyDict[currentPt][region]['idle'] += 1  # not double counting since you still have the vehicle you just changed its state
                newSupplyDict[currentPt][region]['active'] -= 1
            elif element[3] != region:
                newSupplyDict[currentPt][region]['active'] -= 1  # you left the region, remove from active and do not add back to idle, it's lost, will be considered as incoming idle at other region
                newSupplyDict[currentPt][region]['departed'] += 1
                departed+=1 
    
    # ADMISSION CONTROL!!
    # newIRInOut and newBAInOut store the updated InOut dicts where blocked elements are removed, blocking is based on instantaneous supply or projected BA and supply
    newIRInOut=copy.deepcopy(IRInOut)
    newBAInOut=copy.deepcopy(BAInOut)
    # check if you can serve your BA, you only block BAs if there are no idle vehicles (do not have c or more vehicles in the region to handle those that will initiate in the region due to to riders moving across regions and not enough min cost flow updates)
    for element in BAInOut:
        if element[0] == currentPt:  # this means that the vehicles just became active at the current slot within the window
            if (newSupplyDict[currentPt][region]['idle'] == 0):  # if there are no idle veh's
                blocked+=1
                newBAInOut.remove(element)
                print("... WARNING: BA ELEMENT REMOVAL!! the target was not adequately maintained, deviation from target resulted in shortage of drivers beyond BA ...")  # if you maintain ``c" vehicles in region you should not block BA since you do adm. control based on that, but you may have shortage due to less frequent min cost flow balancing and plenty of departures
            else:
                newSupplyDict[currentPt][region]['idle'] -= 1
                newSupplyDict[currentPt][region]['active'] += 1
        # in case of BA you don't update forwardBA since it is already counted in it via BAprofile
        
    # check if you can serve your IR, you block IRs if there's not enough idle or if they interfere with a future BA (fp+fBA+fA) which considers the previously admitted IR and BA in forwardBA
    for element in IRInOut:  # this is going through the InOut that corresponds to requests initiated from the region of interest, note that we only have to worry about current pt as other points are prev. taken care of, either removed from IRInOut or admitted and accounted for as past admits
        if element[0] == currentPt:  # this means that the vehicles just became active at the current slot within the window        
            vehsoujourn = np.arange(element[0], min(winEnd+1, element[1]))  # time points at which vehicle will be active wihthin window
            shouldBlock = False
            if (newSupplyDict[currentPt][region]['idle'] == 0):  # check if there's no available idle
                shouldBlock = True
                blocked+=1
                newIRInOut.remove(element)
                print("... IR blocked, not enough idle ...")
            if shouldBlock is False:  # considering that there are available idle
                for timePt in vehsoujourn:  # check if there is a future violation of BA using the forwardBA
                    if ((forwardBA[timePt]+1)>target):
                        shouldBlock=True
                        blocked+=1
                        newIRInOut.remove(element)
                        print("... IR blocked, interrupts anticipated BA ...")
                        break
            if shouldBlock is False:  # you have enough idle and won't exceed target
                for timePt in vehsoujourn:
                    forwardBA[timePt]+=1  # update the future BA based on it for subsequent elements, note that once we go to the next time point, forwardBA will be automatically updated based on looking at what was already admitted within the time point
                newSupplyDict[currentPt][region]['idle'] -= 1
                newSupplyDict[currentPt][region]['active'] += 1  # if we have enough idling, i.e. you do not go less than zero, which means you had at least 1, then that vehicle becomes active

    return newSupplyDict, blocked, newIRInOut, newBAInOut


def getDelta(regionSupply, targetN):
    '''
    ------------------------------------------------------------------------------------------------------
    computes the deviation from the target for a specific region
    ------------------------------------------------------------------------------------------------------
    '''
    idleInRegion = regionSupply['idle']
    activeInRegion = regionSupply['active']
    delta = targetN - (idleInRegion + activeInRegion)
    return delta, idleInRegion

    
def updateSupply(supplyDict, adjacency, window, updatePt, targetN, tolDef, adjustTotal):
    '''
    ------------------------------------------------------------------------------------------------------
    updates the supply vector using the min-cost flow optimization program
    ------------------------------------------------------------------------------------------------------
    Note: the supply dict MUST HAVE been maintained (maintainSupply) up to the
    updatePt to ensure that the idle and active vehicles at the updatePt
    have been computed so that we know who can we move around and how much is the deficit
    ------------------------------------------------------------------------------------------------------
    tolDef: a tolerance value L, this is set to zero by default, could adjust if we want to limit
    the num. of vehicles moving around
    ------------------------------------------------------------------------------------------------------
    Note: you adjust the supply based on the target n for the *upcoming* window!
    ------------------------------------------------------------------------------------------------------
    Note: supplyDict[updatePt][region]['idle'] = idleVehs[updatePt][region], but we use idleVehs to
    separately keep track of what happens at update points
    ------------------------------------------------------------------------------------------------------
    Note: this method applies over all regions!
    ------------------------------------------------------------------------------------------------------
    '''
    newSupplyDict=copy.deepcopy(supplyDict)
    delta=dict()
    idleVehs=dict()
    delta[updatePt] = dict()
    idleVehs[updatePt] = dict()
    for region in adjacency:  # get the delta per region
        delta[updatePt][region], idleVehs[updatePt][region]=getDelta(supplyDict[updatePt][region], targetN[region][window])
    
    start_nodes, end_nodes, capacities, unit_costs, supplies = createAugmentedNet(updatePt, adjacency, delta, idleVehs, tolDef)
    min_cost = minCostFlow(adjacency, start_nodes, end_nodes, capacities, unit_costs, supplies)
    newSupplyDict = processMinCostResults(min_cost, adjacency, updatePt, newSupplyDict, adjustTotal)  # overwrite the supply at the time point
    return newSupplyDict, delta


def createAugmentedNet(timePt, adjacency, deltas, idleVehs, tolDef):
    '''
   ------------------------------------------------------------------------------------------------------
    creates the augmented network used to run the min cost flow optimization program
    ------------------------------------------------------------------------------------------------------
    '''
    start_nodes = list()
    end_nodes = list()
    capacities = list()
    unit_costs = list()
    supplies = list()
    # create the extra links from auxiliary nodes
    for region in adjacency:
        start_nodes.append(region*100)
        end_nodes.append(region)
        unit_costs.append(1)  # unit cost since we prefer not to send out from a region if we can
        capacities.append(idleVehs[timePt][region])
    # create the actual links (remember they connect to auxiliary nodes)
    for region in adjacency:
        for endregion in adjacency[region]:
            start_nodes.append(region)
            end_nodes.append(endregion*100)
            unit_costs.append(0)
            capacities.append(1000000)
    # create source links, source: 9999
    for region in adjacency:
        start_nodes.append(9999)
        end_nodes.append(region*100)
        unit_costs.append(1000)
        capacities.append(1000000)
    # create the sink links, sink: 6666
    for region in adjacency:
        start_nodes.append(region*100)
        end_nodes.append(6666)
        unit_costs.append(1000)
        capacities.append(1000000)
    # create a direct link from the source to the sink
    start_nodes.append(9999)
    end_nodes.append(6666)
    unit_costs.append(0)
    capacities.append(1000000)
    start_nodes, end_nodes, capacities, unit_costs = zip(*sorted(zip(start_nodes, end_nodes, capacities, unit_costs)))
    start_nodes = list(start_nodes)
    end_nodes = list(end_nodes)
    capacities = list(capacities)
    unit_costs = list(unit_costs)

    
    # first add supplies to transmission nodes
    for region in sorted(adjacency):
        supplies.append(0)  # all actual nodes are transmission nodes
    
    totalSupply=0
    totalDemand=0
    # now go through the auxiliary nodes
    for region in sorted(adjacency):
        if deltas[timePt][region] >= tolDef:  # this means we need vehicles i.e. deficiency or demand node
            supplies.append(-(deltas[timePt][region]-tolDef))  # demand is negative
            totalDemand += deltas[timePt][region]-tolDef
        elif deltas[timePt][region] < tolDef:
            canSupply = min(tolDef-deltas[timePt][region], idleVehs[timePt][region])  # supply is positive
            supplies.append(canSupply)
            totalSupply += canSupply
    
    # now go through sink then source
    supplies.append(-totalSupply)  # sink node, takes the total supply in the network
    supplies.append(totalDemand)  # source node, takes the total demand in the network
    print('supplies: ', supplies )
    return start_nodes, end_nodes, capacities, unit_costs, supplies



def minCostFlow(adjacency, start_nodes, end_nodes, capacities, unit_costs, supplies):
    '''
    ------------------------------------------------------------------------------------------------------
    solves min cost flow using Google OR-tools package
    ------------------------------------------------------------------------------------------------------
    '''
    min_cost = pywrapgraph.SimpleMinCostFlow()  # initiate a min-cost flow optimization program
    links=list()
    for i in range(0, len(start_nodes)):
        min_cost.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])
        links.append((start_nodes[i], end_nodes[i], capacities[i], unit_costs[i]))
    print('links: ', links)
    nodes_list = list()
    for region in sorted(adjacency):
        nodes_list.append(region)
    for region in sorted(adjacency):
        nodes_list.append(region*100)
    nodes_list.append(6666)
    nodes_list.append(9999)
    #print(supplies)
    #print(nodes_list)
    nodesupply=list()
    for i in range(0, len(supplies)):
        min_cost.SetNodeSupply(nodes_list[i], supplies[i])
        nodesupply.append((nodes_list[i], supplies[i]))
    print('node supply: ', nodesupply)
    if min_cost.Solve() == min_cost.OPTIMAL:
        print('Minimum cost:', min_cost.OptimalCost())
        for i in range(min_cost.NumArcs()):
                cost = min_cost.Flow(i) * min_cost.UnitCost(i)
                print('%1s -> %1s   %3s  / %3s       %3s' % (
                  min_cost.Tail(i),
                  min_cost.Head(i),
                  min_cost.Flow(i),
                  min_cost.Capacity(i),
                  cost))

    else:
        print('can not solve')
    return min_cost


def processMinCostResults(min_cost, adjacency, updatePt, supply, adjustTotal):
    '''
    ------------------------------------------------------------------------------------------------------
    processes min cost results by updating the supply dict, brings in new drivers or moves
    eligible drivers around between regions, where the objective is to reduce the deviation
    from the targets across regions
    ------------------------------------------------------------------------------------------------------
    Note:
    adjustTotal: boolean indicating if you want to adjust the total num. of vehicles in the network
    ------------------------------------------------------------------------------------------------------
    '''
    new_supply = copy.deepcopy(supply)  # take in supply and over-write it at update point based on results
    for i in range(min_cost.NumArcs()):  # go through all the arcs
        start = min_cost.Tail(i)  # start node
        end = min_cost.Head(i)  # end node
        flow = min_cost.Flow(i)  # flow
        if adjustTotal is True:
            for region in adjacency:  # go through all the regions to see how to use the arc
                if (start == region) or (start == region*100 and end == 6666):
                    new_supply[updatePt][region]['idle'] -= flow  # you're losing flow to another region or to the sink
                elif (end == region*100):
                    new_supply[updatePt][region]['idle'] += flow  # you're gaining flow either from source or from another region
        elif adjustTotal is False:
            for region in adjacency:  # go through all the regions to see how to use the arc
                if (start == region):
                    new_supply[updatePt][region]['idle'] -= flow  # you're losing flow to another region
                elif (end == region*100 and start != 9999):
                    new_supply[updatePt][region]['idle'] += flow  # you're gaining flow from another region
    return new_supply



def processSupplyBlocked(supply, blocked):
    '''
    ------------------------------------------------------------------------------------------------------
    processes the supply output of the load balancing/adm. control method to return the  percent 
    utilization (idle/total) and the percent of blocked rides (blocked/total)
    ------------------------------------------------------------------------------------------------------
    Note: this function also reorders supply information to be [region][time] instead of [time][region]
    for plotting purposes
    ------------------------------------------------------------------------------------------------------
    '''
    totaltime=list()
    for time in np.arange(1, 36+1, 1):
        totaltime.append(map2time(int(time)))
    efficiency=dict()
    percentBlocked=dict()
    numIdle=dict()
    numBlocked=dict()
    for region in np.arange(1, 4+1, 1):
        efficiency[region]=list()
        percentBlocked[region]=list()
        numIdle[region]=list()
        numBlocked[region]=list()
        for time in np.arange(1, 36+1, 1):
            active=supply[time][region]['active']
            idle=supply[time][region]['idle']
            block=blocked[time][region]
            efficiency[region].append(100*(float(active)/(active+idle)))
            percentBlocked[region].append(100*(float(block)/(active+block)))
            numIdle[region].append(idle)
            numBlocked[region].append(block)
    efficiency['avg']=list()
    percentBlocked['avg']=list()
    numIdle['avg']=list()
    numBlocked['avg']=list()
    for key, time in enumerate(list(np.arange(1, 36+1,1))):
        efficiency['avg'].append(np.mean([efficiency[1][key], efficiency[2][key], efficiency[3][key], efficiency[4][key] ]))
        percentBlocked['avg'].append(np.mean([percentBlocked[1][key], percentBlocked[2][key], percentBlocked[3][key], percentBlocked[4][key] ]))
        numIdle['avg'].append(np.mean([numIdle[1][key], numIdle[2][key], numIdle[3][key], numIdle[4][key] ]))
        numBlocked['avg'].append(np.mean([numBlocked[1][key], numBlocked[2][key], numBlocked[3][key], numBlocked[4][key] ]))
    
    return efficiency, percentBlocked, numIdle, numBlocked, totaltime

    
    

if __name__ == '__main__':
    dDict, head = readCSV('data/ridesLyftMHTN14.csv')
    dDict = addTimeStamp(dDict, slotInMinutes=5)
    dictNum = computeNumberOfUsers(dDict)
    adjacency = {1:[2], 2:[1, 3, 4], 3:[2, 4], 4:[2, 3]}
    initSupply = {1:{'idle':0,'active':0, 'arrived':0, 'departed':0}, 2:{'idle':0,'active':0, 'arrived':0, 'departed':0}, 3:{'idle':0,'active':0, 'arrived':0, 'departed':0}, 4:{'idle':0,'active':0, 'arrived':0, 'departed':0}}  # initial distribution of vehicles
    # get rates
    lamWR = dict()
    lamMintWR = dict()
    dictOSWR = dict()
    for region in adjacency:
        lamWR[region], lamMintWR[region] = getLamPerRegionMLE(dataDict=dDict, Origin=region, slotInMinutes=5, windowInMinutes=20, numWindows=9, Destination=None)
        dictOSWR[region] = getOrderedService(dDict, region=region, slotInMinutes=5, windowInMinutes=20, numWindows=9)

    pBook = 0.0
    tol = 0.01
    sup, bloc, target_n, deltas, sampleProf, ebars, BAprofStore, sampleProfTotal, actualIR, actualProf, IRInOut, BAInOut = manageSupply(lamWR, dictOSWR, dDict, start_slot=1, end_slot=36, window_length=4, percentBooked=pBook, tolerance=tol, initialSupplyDict=initSupply, adjacency=adjacency, tolDef=0, initialN=0)
    efficiency, percentBlocked, numIdle, numBlocked, totaltime = processSupplyBlocked(supply=sup, blocked=bloc)
