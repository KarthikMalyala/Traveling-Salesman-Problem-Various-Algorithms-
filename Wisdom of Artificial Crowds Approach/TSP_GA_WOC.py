# Karthik Malyala
# TSP - Wisdom of Crowds using Genetic Algorithm

# Imports the required libraries
import random
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import perf_counter
from collections import Counter
import math

# Creates a random route (chromosome) with the given cities
def createRandRoute(cities):
    path = random.sample(cities, len(cities))
    path.append(path[0])

    return path

# Creates a population of chromosomes using the createRandRoute function above
def createPopulation(popSize, cities):
    population = []
    for i in range (0, popSize):
        population.append(createRandRoute(cities))
    return population

# Fitness function that determines the total path cost of a chromosome (route)
def fitness(individual):
    totalDistance = 0.0
    for gene in range(0, len(individual) - 1):
       city = individual[gene] # Takes the current city
       nextCity = individual[gene+1] # Takes the next city
       # Assigns the x and y coordinates of both cities for distance calculation
       ax = coordinates[city][0]
       ay = coordinates[city][1]
       bx = coordinates[nextCity][0]
       by = coordinates[nextCity][1]

       # Utilizes the Distance Formula to calculate the distance between the two cities
       pathDistance = (((bx - ax) ** 2) + ((by - ay) ** 2)) ** 0.5
       # The totalDistance is incremented by each path's distance to find the totalDistance of the route in the end
       totalDistance += pathDistance
    return totalDistance

# Ranking function that ranks each individual chromosome in a population by their fitness (cost)
def rankIndividuals(population):
    fitnessDict = {}
    for i in range(0, len(population)):
        routeFit = fitness(population[i])
        fitRatio = 1/routeFit
        fitnessDict[i] = fitRatio
    return sorted(fitnessDict.items(), key = operator.itemgetter(1), reverse = True)

# Selection function that selects individual parents using their fitness as the weight using Roulette Wheel Selection
def selection(rankedPopulation, eliteNum):
    selectRes = []
    df = pd.DataFrame(np.array(rankedPopulation), columns=["Idx", "Cost"])
    df['distSum'] = df.Cost.cumsum()
    df['pick'] = 100 * df.distSum / df.Cost.sum()

    for i in range(0, eliteNum):
        selectRes.append(rankedPopulation[i][0])
    for i in range(0, len(rankedPopulation) - eliteNum):
        pick = 100 * random.random()
        for i in range(0, len(rankedPopulation)):
            if pick <= df.iat[i, 3]:
                selectRes.append(rankedPopulation[i][0])
                break

    return selectRes

# Function that creates a pool of individuals to be paired for reproduction
def matingPool(population, selectRes):
    pool = []
    for i in range(0, len(selectRes)):
        index = selectRes[i]
        pool.append(population[index])
    return pool

# Crossover function that exchanges parts of two single chromosomes to produce a child (new route)
def crossover(parentA, parentB):
    child = []
    childA = []
    childB =[]
    parentA = parentA[:-1]
    parentB = parentB[:-1]
    # print(parentA)
    # print(parentB)
    geneA = int(random.random() * len(parentA))
    geneB = int(random.random() * len(parentA))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childA.append(parentA[i])

    for i in parentB:
        if i not in childA:
            childB.append(i)
    # print('\nchildA: ' + str(childA))
    # print('childB: ' + str(childB))

    child = childA + childB
    if not childA:
        child.append(childB[0])
    else:
        child.append(childA[0])
   # print('Child: '+str(child))
    return child

# Breeding function to create the offspring population while retaining the best routes from the current population and then
# use the crossover function to create children after exchanging parts of chromosomes
def breedPop(matingPool, eliteNum):
    children = []
    poolLen = len(matingPool) - eliteNum
    samplePool = random.sample(matingPool, len(matingPool))

    for i in range (0, eliteNum):
        children.append(matingPool[i])

    for i in range(0, poolLen):
        child = crossover(samplePool[i], samplePool[len(matingPool) - i - 1])
        children.append(child)
    return children

# Mutation function that uses the mutation rate to swap two cities within an individual chromosome (route)
def mutate(route, mRate):
    for city in range(len(route)):
        if(random.random() < mRate):
            #print("\nMutation Occurs\n")
            swapCity = int(random.random() * len(route))

            cityA = route[city]
            cityB = route[swapCity]

            route[city] = cityB
            route[swapCity] = cityA

            startCity = route[-1]

            if (startCity != route[0]):
                if (route.count(route[0]) > 1):
                    route.pop(0)
                    route.insert(0, startCity)
                else:
                    route.pop()
                    route.append(route[0])
    return route

# Function that uses mutation to mutate through every chromosome in a given population
def mutatePop(population, mRate):
    mutatedPopulation = []

    for individual in range(0, len(population)):
        mutatedRoute = mutate(population[individual], mRate)
        mutatedPopulation.append(mutatedRoute)
    return mutatedPopulation

# Function that creates the next generation using the current generation to call all the functions defined above
def newGen(curGen, eliteNum, mRate):
    rankedPopulation = rankIndividuals(curGen)
    selectRes = selection(rankedPopulation, eliteNum)
    matingpool = matingPool(curGen, selectRes)
    children = breedPop(matingpool, eliteNum)
    newGen = mutatePop(children, mRate)
    return newGen

# Average function that returns the average of a given list
def average(progress, listIdx):
    testList = progress[listIdx:(listIdx + 500)]
    return sum(testList)/len(testList)

# Graphing function that graphs both the results of the improvement curve as well as the route diagram in a optimized route fashion with two
# subplots using matplotlib
def graphPlot(progress, listIdx, route, popSize, eliteNum, mutationRate, generations, run):
    graph = plt.figure(figsize=(100, 100))
    # Sets a subplot to the left for the improvement curve
    costGen = graph.add_subplot(1, 2, 1)
    # Sets a subplot to the right for the optimized route
    cityPlot = graph.add_subplot(1, 2, 2)

    # Initally empty lists for coordinates to be added for every city
    ox = []
    oy = []

    # Gets the coordinates of each city of the new optimized route (bestRoute)
    for i in range(len(route)):
        city = route[i]
        ox.append(coordinates[city][0])
        oy.append(coordinates[city][1])

    ox.append(ox[0])
    oy.append(oy[0])

    routeDist = fitness(route)

    # Sets titles and labels for the plots and corresponding axes
    graph.suptitle('TSP Genetic Algorithm (RUN ' + str(run) + '): Population(' + str(popSize) + ') EliteNum(' + str(eliteNum) + ') MutationRate(' + str(mutationRate) + ') Generations(' +
                   str(listIdx + 1) + '), Distance(' + str(routeDist) + ')')
    costGen.set_title('Cost vs. Generation', weight='bold')
    cityPlot.set_title('Optimized Route', weight='bold')
    costGen.set_xlabel('Generation')
    costGen.set_ylabel('Cost')
    cityPlot.set_xlabel('X-Coordinate')
    cityPlot.set_ylabel('Y-Coordinate')

    costGen.plot(progress)

    # Plots each city according to its coordinates and connects cities with arrows, according to the order of the original route
    for i in range(0, cityCount):
        cityPlot.scatter(ox[i], oy[i], c='red', marker='o',
                         s=20)  # Plots city coordinates with a circle marker and red color
        cityLbl = ""
        # Finds what city the coordinates plotted above correspond to
        for city, coords in coordinates.items():
            if coords == (ox[i], oy[i]):
                cityLbl += str(city)
        cityPlot.annotate(cityLbl, (ox[i], oy[i]),
                          weight='bold')  # Annotates each point with a label of what city the point is representing
        cityPlot.scatter(ox[i + 1], oy[i + 1], c='blue', marker='o',
                         s=20)  # Labels the starting city with a blue circle marker
        # Connects every city with an arrow according to the order of the original route
        cityPlot.quiver(ox[i], oy[i], (ox[i + 1] - ox[i]), (oy[i + 1] - oy[i]), angles='xy', scale_units='xy', scale=1)

    plt.show()

# Overall genetic algorithm function that takes progress of the cost for the improvement curve and generates a new population
# over time to see the difference between the cost and generation when iterating through
def geneticAlgorithm(cities, popSize, eliteNum, mutationRate, generations, run):

    startTime = perf_counter()
    population = createPopulation(popSize, cities)
    progress = []
    curCost = 0.0
    counter = 0
    listIdx = 0
    oldAvg = 0.0
    avgByFar = 0.0
    progress.append(1 / rankIndividuals(population)[0][1])
    print("Initial Dist: " + str(1/(rankIndividuals(population)[0][1])))

    for i in range(0, generations):
        population = newGen(population, eliteNum, mutationRate)
        print(listIdx / generations)
        #print(rankIndividuals(population)[0][1])
        print("Generation: " + str(listIdx) + " Cost: " + str(1 / rankIndividuals(population)[0][1]))
        progress.append(1 / rankIndividuals(population)[0][1])
        counter += 1

        routeIdx = rankIndividuals(population)[0][0]
        route = population[routeIdx]
        #print(route)

        if (counter == 500):
            # graphPlot(progress, listIdx, route)
            # plt.clf()
            if (listIdx == 500):
                oldAvg = average(progress, 0)
                counter = 0
            else:
                avgByFar = average(progress, listIdx)
                if (abs(avgByFar - oldAvg) < 20):
                    print("Average is too small")
                    break
                else:
                    oldAvg = average(progress, listIdx)
                    counter = 0
        listIdx += 1
    print("Final Dist: " + str(1 / (rankIndividuals(population)[0][1])))
    print('Algorithm Runtime: ' +  str(perf_counter() - startTime))
    #finalEdges = WOC(population)
    print(route)
    #print(finalEdges)
    # Plots the final route
    graphPlot(progress, listIdx, route, popSize, eliteNum, mutationRate, generations, run)
    #bestRouteIdx = rankIndividuals(population)[0][0]
    #bestRoute = population[bestRouteIdx]
    expertRoutes = []
    # In the first two runs (less population size), only return top 5 fittest individuals
    if(run == 1 or run == 2):
        for index in range(0, 6):
            routeIdx = rankIndividuals(population)[index][0]
            curRoute = population[routeIdx]
            expertRoutes.append(curRoute)
    # In last two runs (larger population size), only return top 15 fittest individuals
    elif(run == 3 or run == 4):
        for index in range(0, 16):
            routeIdx = rankIndividuals(population)[index][0]
            curRoute = population[routeIdx]
            expertRoutes.append(curRoute)

    print(expertRoutes)
    # Returns the fittest routes in the given run
    return expertRoutes

# Function that calculates the distance between two given cities using the distance formula
def cityDistance(city, testCity):
    # Grabs the coordinates of each city
    ax = coordinates[city][0]
    ay = coordinates[city][1]
    bx = coordinates[testCity][0]
    by = coordinates[testCity][1]

    # Uses the distance formula
    distance = (((bx - ax) ** 2) + ((by - ay) ** 2)) ** 0.5
    return distance

# Function that calculates the angles for each of the cities in a given line segment when extended towards a test city by using Law of Cosines
def calcAngles(cityA, cityB, testCity):
    # Calculates the sides of the potential triangle with distance formula
    sideA = cityDistance(cityB, testCity)
    sideB = cityDistance(cityA, testCity)
    sideC = cityDistance(cityA, cityB)

    # Utilizes the Law of Cosines, we can calculate angle AC (angle between edge AB and potential edge AC)
    thetaAC = math.degrees(math.acos(((sideB * sideB) + (sideC * sideC) - (sideA * sideA)) / (2 * sideB * sideC)))
    # Utilizes the Law of Cosines, we can calculate angle BC (angle between edge AB and potential edge BC)
    thetaBC = math.degrees(math.acos(((sideC * sideC) + (sideA * sideA) - (sideB * sideB)) / (2 * sideA * sideC)))

    # Returns the angles found above
    return thetaAC, thetaBC

# Function that uses the angles found from the above function and a line segment to measure the perpendicular distance from the node to the line segment using sin
def nodeDistance(thetaAT, thetaBT, cityA, cityB, testCity):
    # Calculates the two sides that go to the testCity from the given line segment
    sideA = cityDistance(cityA, testCity)
    sideB = cityDistance(cityB, testCity)

    # Calculates the distance between the testCity and the line segment using sin (SOH CAH TOA where Sin is Opposite over Hypotenuese)
    # Since we know the hypotenuse through the sides calculated above, we can find the opposite length which gives us the nodeDistance
    distance1 = sideA * math.sin(math.radians(thetaAT))
    distance2 = sideB * math.sin(math.radians(thetaBT))

    return distance1

# Function that deals with a special edge case wherein the distance between two unique segments and a common node is the same by calculating the shortest angle between them two
def angleFromEdge(cityA, cityB, testCity, point):
    # If the intersecting node for the two unique segments is at cityA
    if(point == 'a'):
        # Calculate the sides for to find the total angle rotation from the line segment to the potential edge to the new node
        sideA = cityDistance(cityB, testCity)
        sideB = cityDistance(cityA, testCity)
        sideC = cityDistance(cityA, cityB)

        # Use Law of Cosines once again to find the angle between the given line segment and the new potential edge
        # Calculates Angle AT
        angle = math.degrees(math.acos(((sideB * sideB) + (sideC * sideC) - (sideA * sideA)) / (2 * sideB * sideC)))

    # If the intersecting node for the two unique segments is at cityB
    elif(point == 'b'):
        # Calculate the sides for to find the total angle rotation from the line segment to the potential edge to the new node
        sideA = cityDistance(cityB, testCity)
        sideB = cityDistance(cityA, testCity)
        sideC = cityDistance(cityA, cityB)

        # Use Law of Cosines once again to find the angle between the given line segment and the new potential edge
        # Calulcates Angle BT
        angle = math.degrees(math.acos(((sideA * sideA) + (sideC * sideC) - (sideB * sideB)) / (2 * sideA * sideC)))

    # Returns the found angle
    return angle

# Distance function that calculates the total distance (cost) of the provided edges
def totalDistance(visitedEdges):
    totalDistance = 0.0
    for edge in range(len(visitedEdges)):
       #print(edge)
       city = visitedEdges[edge][0] # Takes the current city
       nextCity = visitedEdges[edge][1] # Takes the next city
       # Assigns the x and y coordinates of both cities for distance calculation
       ax = coordinates[city][0]
       ay = coordinates[city][1]
       bx = coordinates[nextCity][0]
       by = coordinates[nextCity][1]

       # Utilizes the Distance Formula to calculate the distance between the two cities
       pathDistance = (((bx - ax) ** 2) + ((by - ay) ** 2)) ** 0.5
       # The totalDistance is incremented by each path's distance to find the totalDistance of the route in the end
       totalDistance += pathDistance
    return totalDistance

# Finds the edges in the given path
def findEdges(list):
    edges = []
    for city in range(len(list) - 1):
        nodeA = list[city]
        nodeB = list[city + 1]
        edges.append([nodeA, nodeB])
    return edges

# Function that counts the number of occurences for each edge provided in a list
def countDups(masterEdges):
    duplicate = []
    c = Counter(map(tuple, masterEdges))
    for k,v in c.items():
        if (v > 1):
            duplicate.append([k,v])
    duplicate.sort(key=operator.itemgetter(1), reverse= True)

    return duplicate

# Wisdom of Crowds Algorithm with Aggregation
def WOC(population):
    masterEdges = []
    # Finds the edges in every path of the population
    for path in population:
        pathEdges = findEdges(path)
        masterEdges.extend(pathEdges)
    #print(str(masterEdges))
    # Removes reversed duplicates of edges
    finalPairs = [tuple(item) for item in map(sorted, masterEdges)]
    # Finds the number of occurrences for each edge
    dups = countDups(finalPairs)
    #print('Dups: ' + str(dups))
    bestEdges = []
    copyEdges = []
    for edge in dups:
        copyEdges.append(edge[0])
    #print(str(copyEdges))
    visitedCities = []
    startEdge = copyEdges[0]
    checkedEdges = []
    bestEdges.append(startEdge)
    visitedCities.append(startEdge[0])
    visitedCities.append(startEdge[1])
    copyEdges.remove(startEdge)
    checkedEdges.append(startEdge)

    # Repetitively iterates through unchecked edges to find connections and builds an optimal path
    restart = 1
    while(restart == 1):
        for edge in copyEdges:
            if(edge not in checkedEdges):
                if(edge[0] == startEdge[1]):
                    if edge[1] not in visitedCities:
                        bestEdges.append(edge)
                        visitedCities.append(edge[1])
                        startEdge = edge
                        checkedEdges.append(edge)
                        copyEdges.remove(edge)
                        restart = 1
                        break
                elif(edge[1] == startEdge[1]):
                    if edge[0] not in visitedCities:
                        bestEdges.append(tuple(reversed(edge)))
                        visitedCities.append(edge[0])
                        startEdge = tuple(reversed(edge))
                        checkedEdges.append(edge)
                        copyEdges.remove(edge)
                        restart = 1
                        break
            if(edge[0] in visitedCities and edge[1] in visitedCities):
                restart = 0

        # print('Checked: ' + str(bestEdges))
        # print(str(copyEdges))
        # print(str(visitedCities))

    # print(visitedCities)
    # Returns the bestEdges and visitedCities of the computed optimal path
    return bestEdges, visitedCities

# Greedy algorithm that uses closest edge insertion heursitic to form a valid TSP route if the aggregation method did not cover all cities
def greedy(bestEdges, cities, citiesChecked):
    counter = 0  # Counter variable that's used to keep track of the while loop
    restart = 1  # Variable that will turn 0 when the list of unvisited cities becomes null which ultimately breaks the while loop

    while restart == 1:
        # Variables that keep track of the shortest distance to node
        shortestDist = 0
        # The best node that's closest to an edge that has been previously formed
        bestTarget = 0
        # Two cities that signify what line segment is to be broken in order to connect the best node
        city1 = 0
        city2 = 0

        completeCheck = all(city in citiesChecked for city in cities)

        if completeCheck is True:
            #citiesChecked.append(citiesChecked[0])
            #print('Included')
            break

        # For loop that iterates through every edge that has been formed already
        for pair in bestEdges:
            # Gets the two cities that form the edge
            cityA = pair[0]
            cityB = pair[1]
            # Since the two cities form an existing edge, remove them from the list of unvisited cities
            if cityA in cities:
                cities.remove(cityA)
            if cityB in cities:
                cities.remove(cityB)
            # Nested for loop that goes through every city within the list of unvisited cities and compares its distance to existing edges
            for testCity in cities:
                # Calculates the two angles from the existing edge to the test city using Law of Cosines
                thetaAT, thetaBT = calcAngles(cityA, cityB, testCity)
                # If both angles are acute, then calculate the distance of the test city to the edge using sin in SOH CAH TOA
                if (thetaAT < 90 and thetaBT < 90):
                    distance = nodeDistance(thetaAT, thetaBT, cityA, cityB, testCity)
                # If the angle from cityA to testCity is greater than 90, then calculate the linear distance from cityA to testCity
                elif (thetaAT >= 90):
                    distance = cityDistance(cityA, testCity)
                    # Point variable to denote where the angle pivots from in terms of city
                    point = 'a'
                # If the angle from cityB to testCity is greater than 90, then calculate the linear distance from cityB to testCity
                elif (thetaBT >= 90):
                    distance = cityDistance(cityB, testCity)
                    # Point variable to denote where the angle pivots from in terms of city
                    point = 'b'

                # If no shortestDist has been recorded yet, assign the first iteration's distance to do so
                if (shortestDist == 0):
                    shortestDist = distance
                    bestTarget = testCity
                    city1 = cityA
                    city2 = cityB

                # If the current iteration's distance is lesser than the shortestDist by far, update it with the current distance
                elif (distance < shortestDist):
                    shortestDist = distance
                    bestTarget = testCity
                    city1 = cityA
                    city2 = cityB

                # Edge case that solves when two edges are the same distance to a certain test city
                elif (distance == shortestDist):
                    # Assigns the old pivot point for angle calculation
                    if (city1 == cityA) or (city1 == cityB):
                        oldPoint = 'a'
                    if (city2 == cityA) or (city2 == cityB):
                        oldPoint = 'b'
                    # Calculates the angle from the edge to the segment that would connect the pivot city to the target city for both old and new cases
                    angleOld = angleFromEdge(city1, city2, testCity, oldPoint)
                    angleNew = angleFromEdge(cityA, cityB, testCity, point)

                    # If the new angle between the new pivot point and the testCity is lesser than the old angle, update the shortestDist and cities
                    if (angleNew < angleOld):
                        shortestDist = distance
                        bestTarget = testCity
                        city1 = cityA
                        city2 = cityB

        # After all the iterations, delete the best target found from the list of unvisited cities
        if bestTarget in cities:
            cities.remove(bestTarget)
        # Since the edge between the two cities being chosen is being broken apart, remove it from the list of visitedEdges
        if [city1, city2] in bestEdges and counter >= 1:
            bestEdges.remove([city1, city2])
        if [city1, bestTarget] not in bestEdges:
            bestEdges.append([city1, bestTarget])
        if [city2, bestTarget] not in bestEdges:
            bestEdges.append([city2, bestTarget])
        # Calculates the total cost of the route established with edges by far
        if bestTarget not in citiesChecked:
            citiesChecked.append(bestTarget)
        # plotGraph(visitedEdges, coordinates, cityCount, totalDist)
        # Increments the counter
        counter += 1
        # If there are no more unvisited cities left, break the while loop
        if len(cities) == 0:
            restart = 0
        #citiesChecked.append(citiesChecked[0])
    return(bestEdges, citiesChecked)

# Function that plots the given route from WOC above along with the GA run results in a bar plot
def WOCPlot(route, coordinates, cityCount, runs, costs):
    graph = plt.figure(figsize=(100, 100))
    # Sets a subplot to the left for the improvement curve
    gaRuns = graph.add_subplot(1, 2, 1)
    # Sets a subplot to the right for the optimized route
    cityPlot = graph.add_subplot(1, 2, 2)

    gaRuns.bar(runs, costs, width = 0.2, color = ['black', 'red', 'black', 'red'])
    #cityPlot = plt.figure()

    ox = []
    oy = []

    # Gets the coordinates of each city of the new optimized route (bestRoute)
    for i in range(len(route)):
        city = route[i]
        ox.append(coordinates[city][0])
        oy.append(coordinates[city][1])

    ox.append(ox[0])
    oy.append(oy[0])

    route.append(route[0])
    distance = fitness(route)

    # Plots each city according to its coordinates and connects cities with arrows, according to the order of the original route
    for i in range(0, cityCount):
        cityPlot.scatter(ox[i], oy[i], c='red', marker='o',
                         s=20)  # Plots city coordinates with a circle marker and red color
        cityLbl = ""
        # Finds what city the coordinates plotted above correspond to
        for city, coords in coordinates.items():
            if coords == (ox[i], oy[i]):
                cityLbl += str(city)
        cityPlot.annotate(cityLbl, (ox[i], oy[i]),
                          weight='bold')  # Annotates each point with a label of what city the point is representing
        cityPlot.scatter(ox[i + 1], oy[i + 1], c='blue', marker='o',
                         s=20)  # Labels the starting city with a blue circle marker
        # Connects every city with an arrow according to the order of the original route
        cityPlot.quiver(ox[i], oy[i], (ox[i + 1] - ox[i]), (oy[i + 1] - oy[i]), angles='xy', scale_units='xy', scale=1)

    # Sets titles and labels for the plots and corresponding axes
    graph.suptitle('TSP GA + WOC Results: Optimal WOC Solution = ' + str(distance))
    gaRuns.set_title('GA Costs over 4 Runs', weight='bold')
    cityPlot.set_title('Optimized Route', weight='bold')
    gaRuns.set_xlabel('Run')
    gaRuns.set_ylabel('Cost')
    cityPlot.set_xlabel('X-Coordinate')
    cityPlot.set_ylabel('Y-Coordinate')

    plt.show()

# Main driver function
if __name__ == '__main__':
    # Assigns the specified file to a variable
    file = r"Random222.tsp"
    coordinates = {}  # Stores the cities and their corresponding coordinates in a dictionary

    # Opens the file and reads line-by-line after 'NODE_COORD_SECTION' to get coordinates for each city
    with open(file) as inp:
        for line in inp:
            if 'NODE_COORD_SECTION' in line:
                for line in inp:
                    city = line.rstrip('\n')
                    coord = city.split(' ')
                    index = int(coord[0])  # Finds the City
                    x = float(coord[1])  # Finds its X-Coordinate
                    y = float(coord[2])  # Finds its Y-Coordinate
                    coordinates[index] = (x, y)  # Assign each city with a tuple of its coordinates in the dictionary
            # If we reached the end of the file then break out of the loop
            elif 'EOF' in line:
                break

    # Makes a list of cities to keep track of what cities have not been visited
    cities = []
    for city in coordinates:
        cities.append(city)

    # Gets the number of cities in the given file
    cityCount = len(coordinates)

    counter = 1
    run1 = geneticAlgorithm(cities = cities, popSize= 40, eliteNum = 8, mutationRate= 0.0002, generations= 10000, run = counter)
    counter += 1
    run2 = geneticAlgorithm(cities = cities, popSize= 60, eliteNum = 12, mutationRate= 0.0003, generations= 10000, run = counter)
    counter += 1
    run3 = geneticAlgorithm(cities = cities, popSize= 80, eliteNum = 16, mutationRate= 0.0004, generations= 10000, run = counter)
    counter += 1
    run4 = geneticAlgorithm(cities = cities, popSize= 100, eliteNum = 20, mutationRate= 0.0005, generations= 10000, run = counter)

    print('Run 1: ' + str(run1))
    print('Run 2: ' + str(run2))
    print('Run 3: ' + str(run3))
    print('Run 4: ' + str(run4))
    experts = []
    experts.extend(run1)
    experts.extend(run2)
    experts.extend(run3)
    experts.extend(run4)

    run1Fit = fitness(run1[0])
    run2Fit = fitness(run2[0])
    run3Fit = fitness(run3[0])
    run4Fit = fitness(run4[0])

    print('\nGA Run 1: ' + str(run1Fit))
    print('GA Run 2: ' + str(run2Fit))
    print('GA Run 3: ' + str(run3Fit))
    print('GA Run 4: ' + str(run4Fit))
    # print(str(experts))
    # print('Cities : '+ str(cities))
    bestEdges, citiesChecked = WOC(experts)
    # print('BestEdges: ' + str(bestEdges))
    # print('Cities Checked: ' + str(citiesChecked))
    bestPath, finalCities = greedy(bestEdges, cities, citiesChecked)
    finalCities.append(finalCities[0])

    print('\nBest Path: ' + str(finalCities))
    #print('Best Path: ' + str(bestPath))
    finalDistance = fitness(finalCities)

    runs = ['1', '2', '3', '4']
    costs = [run1Fit, run2Fit, run3Fit, run4Fit]

    WOCPlot(finalCities, coordinates, cityCount, runs, costs)



