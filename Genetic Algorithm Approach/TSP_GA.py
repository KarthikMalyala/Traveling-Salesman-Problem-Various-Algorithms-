# Karthik Malyala
# TSP - Genetic Algorithm

# Imports the required libraries
import random
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Creates a random route (chromosome) with the given cities
def createRandRoute(cities):
    path = random.sample(cities, len(cities))
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

    geneA = int(random.random() * len(parentA))
    geneB = int(random.random() * len(parentA))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childA.append(parentA[i])

    for i in parentB:
        if i not in childA:
            childB.append(i)

    child = childA + childB
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
            swapCity = int(random.random() * len(route))

            cityA = route[city]
            cityB = route[swapCity]

            route[city] = cityB
            route[swapCity] = cityA
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
def graphPlot(progress, listIdx, route):
    graph = plt.figure(figsize=(100, 100))
    # Sets a subplot to the left for the improvement curve
    costGen = graph.add_subplot(1, 2, 1)
    # Sets a subplot to the right for the optimized route
    cityPlot = graph.add_subplot(1, 2, 2)

    # Sets titles and labels for the plots and corresponding axes
    graph.suptitle('TSP Genetic Algorithm: Population(100), EliteNum(10), MutationRate(0.001), Generations(' + str(
                listIdx + 1) + ')')
    costGen.set_title('Cost vs. Generation', weight='bold')
    cityPlot.set_title('Optimized Route', weight='bold')
    costGen.set_xlabel('Generation')
    costGen.set_ylabel('Cost')
    cityPlot.set_xlabel('X-Coordinate')
    cityPlot.set_ylabel('Y-Coordinate')

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
def geneticAlgorithm(cities, popSize, eliteNum, mutationRate, generations):
    population = createPopulation(popSize, cities)
    progress = []
    curCost = 0.0
    counter = 0
    listIdx = 0
    oldAvg = 0.0
    avgByFar = 0.0
    progress.append(1 / rankIndividuals(population)[0][1])
    print("Initial Dist: " + str(1/rankIndividuals(population)[0][1]))

    for i in range(0, generations):
        population = newGen(population, eliteNum, mutationRate)
        print(listIdx / generations)
        print("Generation: " + str(listIdx) + " Cost: " + str(1 / rankIndividuals(population)[0][1]))
        progress.append(1 / rankIndividuals(population)[0][1])
        counter += 1

        routeIdx = rankIndividuals(population)[0][0]
        route = population[routeIdx]

        # Stopping criteria for every 500 generations
        if (counter == 500):
            graphPlot(progress, listIdx, route)
            plt.clf()
            if (listIdx == 500):
                oldAvg = average(progress, 0)
                counter = 0
            else:
                avgByFar = average(progress, listIdx)
                if (abs(avgByFar - oldAvg) < 20):
                    print("Avg too small")
                    break
                else:
                    oldAvg = average(progress, listIdx)
                    counter = 0
        listIdx += 1

    print("Final Dist: " + str(1 / rankIndividuals(population)[0][1]))
    bestRouteIdx = rankIndividuals(population)[0][0]
    bestRoute = population[bestRouteIdx]

    # Returns the best route found
    return bestRoute

# Main driver function
if __name__ == '__main__':
    # Assigns the specified file to a variable
    file = r"C:\Fall 2021\CSE 545\Project 4\Random100.tsp"
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

    visitedEdges = [] # List to keep track of the edges that have been formed by far
    citiesChecked = [] # List to keep track of all the cities that have been picked
    # Gets the number of cities in the given file
    cityCount = len(coordinates)

    # Performs the genetic algorithm on the given 100 city TSP file with:
    # Population Size = 100
    # Elite Number = 10
    # Mutation Rate = 0.001
    # Generations = 8000 (max value)
    geneticAlgorithm(cities = cities, popSize= 100, eliteNum = 10, mutationRate= 0.001, generations= 8000)