# Karthik Malyala
# TSP - Closest Edge Insertion

# Imports the required libraries
import matplotlib.pyplot as plt
import math
from time import perf_counter

# Function that determines the first closest node to a given starCity to form the first edge
def startingPair(coordinates, startCity):
    shortestDist = 0
    # Grabs the coordinates of the starting city
    ax = coordinates[startCity][0]
    ay = coordinates[startCity][1]

    # Iterates through each city in the given coordinates and records the shortest distance by far
    for city in coordinates:
        if city == startCity:
            continue
        bx = coordinates[city][0]
        by = coordinates[city][1]
        # Calculates the distance between two nodes using the distance formula
        nodeDistance = (((bx - ax) ** 2) + ((by - ay) ** 2)) ** 0.5
        # Keeps track of the shortest distance and best node by far
        if shortestDist == 0:
            shortestDist = nodeDistance
            bestNode = city
        if nodeDistance < shortestDist:
            shortestDist = nodeDistance
            bestNode = city

    # Returns the best node (closest) found
    return bestNode


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

# Function that plots the given edges by far onto a graph to show the closest edge insertion heuristic
def plotGraph(visitedEdges, coordinates, cityCount, totalDist):
    # print(cityCount)
    x = []
    y = []
    # Gets the coordinates of each city in the original route
    for i in range(1, cityCount + 1):
        city = i
        x.append(coordinates[city][0])
        y.append(coordinates[city][1])

    for i in range(0, cityCount):
        plt.scatter(x[i], y[i], c='red', marker='o', s=40)  # Plots city coordinates with a circle marker and red color
        cityLbl = "City "
        # Finds what city the coordinates plotted above correspond to
        for city, coords in coordinates.items():
            if coords == (x[i], y[i]):
                cityLbl += str(city)
        plt.annotate(cityLbl, (x[i], y[i]), weight='bold')  # Annotates each point with a label of what city the point is representing

    nx = []
    ny = []
    for i in range(len(visitedEdges)):
        for j in visitedEdges[i]:
            nx.append(coordinates[j][0])
            ny.append(coordinates[j][1])

    # Outputs the data collected by far and isolates the solution in a different color
    for i in range(0, len(nx), 2):
        plt.plot(nx[i:i+2], ny[i:i+2])
    #plt.title("30 Cities with startCity 1\nTotal Distance: " + str(totalDist))
    plt.title("40 Cities with startCity 7\nTotal Distance: " + str(totalDist))
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
    plt.show()

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


if __name__ == '__main__':
    # Assigns the specified file to a variable
    file = r"C:\Fall 2021\CSE 545\Project 3\Random40.tsp"
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
    #### DETERMINES THE STARTING CITY #######
    startCity = 7
    # Gets the nearest city to the starting city
    closestCity = startingPair(coordinates, startCity)
    # Forms an edge between the startCity and closestCity and appends it to the list of visitedEdges
    visitedEdges.append([startCity, closestCity])
    citiesChecked.append(startCity)
    citiesChecked.append(closestCity)

    counter = 0 # Counter variable that's used to keep track of the while loop
    restart = 1 # Variable that will turn 0 when the list of unvisited cities becomes null which ultimately breaks the while loop
    startTime = perf_counter() # Starts the timer for algorithm runtime
    while restart == 1:
        # Variables that keep track of the shortest distance to node
        shortestDist = 0
        # The best node that's closest to an edge that has been previously formed
        bestTarget = 0
        # Two cities that signify what line segment is to be broken in order to connect the best node
        city1 = 0
        city2 = 0

        # For loop that iterates through every edge that has been formed already
        for pair in visitedEdges:
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
                elif(thetaBT >= 90):
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
        if [city1, city2] in visitedEdges and counter >= 1:
            visitedEdges.remove([city1, city2])
        if [city1, bestTarget] not in visitedEdges:
            visitedEdges.append([city1, bestTarget])
        if [city2, bestTarget] not in visitedEdges:
            visitedEdges.append([city2, bestTarget])
        # Calculates the total cost of the route established with edges by far
        citiesChecked.append(bestTarget)
        #plotGraph(visitedEdges, coordinates, cityCount, totalDist)
        # Increments the counter
        counter += 1
        # If there are no more unvisited cities left, break the while loop
        if len(cities) == 0:
            restart = 0

    # Calculating the total algorithm runtime above
    endTime = perf_counter()
    runtime = endTime - startTime
    print("\n#### 40 CITIES (startCity 7) ####")
    print("Order of Nodes Inserted: " + str(citiesChecked))
    totalDist = totalDistance(visitedEdges)
    print("Closest Edge Insertion Heuristic Runtime: " + str(runtime))
    print("Total Cost for 40 Cities: " + str(totalDist))
    #plotGraph(visitedEdges, coordinates, cityCount, totalDist)