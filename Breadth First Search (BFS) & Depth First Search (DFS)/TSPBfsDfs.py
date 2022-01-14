# Karthik Malyala
# TSP - BFS & DFS

# Imports the required libraries
import matplotlib.pyplot as plt
from time import perf_counter

# Hardcoded the given adjacency matrix into a dictionary
euclidianPaths = {1: [2,3,4],
                  2: [3],
                  3: [4,5],
                  4: [5,6,7],
                  5: [7,8],
                  6: [8],
                  7: [9,10],
                  8: [9,10,11],
                  9: [11],
                  10: [11],
                  11: []}

# Distance function that calculates the total distance (cost) of a path
def distance(path):
    totalDistance = 0
    if (len(path) == 1):
        return(totalDistance)
    elif(len(path) > 1):
        for x in range(0, len(path) -1):
           city = path[x] # Takes the current city
           nextCity = path[x + 1] # Takes the next city
           # Assigns the x and y coordinates of both cities for distance calculation
           ax = coordinates[city][0]
           ay = coordinates[city][1]
           bx = coordinates[nextCity][0]
           by = coordinates[nextCity][1]

           # Utilizes the Distance Formula to calculate the distance between the two cities
           pathDistance = (((bx - ax) ** 2) + ((by - ay) ** 2)) ** 0.5
           # The totalDistance is incremented by each path's distance to find the totalDistance of the route in the end
           totalDistance += pathDistance
    return(totalDistance)

# First Implementation of BFS which returns the path with the Least Amount of Nodes to the target city
def BFS(matrix, startCity, endCity):
    startTime = perf_counter()
    queue = [[startCity]] # Utilizes the queue data structure
    visited_nodes = [] # Keeps track of the nodes that have been visited already
    smallestCost = 0  # Initial smallestCost
    bestRoute = [] # Empty bestRoute that will be overwritten with best route

    while len(queue) > 0:
        route = queue.pop(0) # FIFO Traversal
        curNode = route[-1]
        # If the current node is already visited, don't traverse through it
        if curNode not in visited_nodes:
            # Get all the same-level nodes
            adj_nodes = matrix[curNode]
            # For every adjacent node, append it to the path and queue and see if it has reached target city yet
            for item in adj_nodes:
                path = list(route)
                path.append(item)
                queue.append(path)
                if item == endCity:
                    #print(path)
                    pathDistance = distance(path)
                    # Keeps track of the smallest cost and best route
                    if smallestCost == 0.0:
                        smallestCost = pathDistance
                        bestRoute = path
                    elif smallestCost > pathDistance:
                        smallestCost = pathDistance
                        bestRoute = path

                visited_nodes.append(curNode) # Appends the current node to the visited list so it won't be traversed through again
    return (bestRoute, smallestCost, (perf_counter() - startTime))

# Second Implementation of BFS that returns the Cheapest Route possible to the Target City
def CheapestBFS(matrix, start, endCity):
    startTime = perf_counter()
    smallestCost = 0
    bestRoute = []
    pathToGetThere = [start] # A list of nodes to describe the backtracking algorithm paradigm and to determine the cheapest path
    queue = [(start, pathToGetThere)] # A modified version of the queue to incorporate backtracking
    while queue:
        #print (queue)
        (city, path) = queue.pop(0) # FIFO Traversal
        for next in matrix[city]:
            if next == endCity:
                # Add the target city to the possible solution
                solution = path + [next]
                pathDistance = distance(solution)
                # Keep track of the smallest cost and best route
                if smallestCost == 0:
                    smallestCost = pathDistance
                    bestRoute = solution
                elif pathDistance < smallestCost:
                    smallestCost = pathDistance
                    bestRoute = solution
            else:
                # Append the queue if endCity is not met
                queue.append((next, path + [next]))

    return bestRoute, smallestCost, (perf_counter() - startTime)

# First Implementation of DFS that returns the first deepest path to the target city (recursive)
def DFS(stack):
    startTime = perf_counter()
    global visitedNodes, euclidianPaths # Uses global variables that have been predefined outside the function to not mess with the recursion
    # If the current path has already been visited, pop the stack
    if set(euclidianPaths[stack[-1]]).issubset(visitedNodes):
        del stack[-1]
        return DFS(stack) # Recursive step

    # If not, proceed to traverse through each node that its connected to
    for item in euclidianPaths[stack[-1]]: # LIFO Traversal
        if item in visitedNodes:
            continue
        visitedNodes.append(item)
        stack.append(item)
        # If endCity is met, calculate the distance and return. (Doesn't care about the # of nodes or distance)
        if item == 11:
            distancePath = distance(stack)
            return stack, distancePath, (perf_counter() - startTime)
        else:
            return DFS(stack) # Recursive Step

# Second Implementation of DFS that returns the Cheapest Route possible to the target city (Iterative)
def CheapestDFS(matrix, startCity, endCity):
    startTime = perf_counter()
    smallestCost = 0
    bestRoute = []
    stack = [(startCity, [startCity])] # For backtracking
    while stack:
        #print(stack)
        (city, path) = stack.pop() # LIFO Traversal
        for neighbor in matrix[city]:
            if neighbor == endCity:
                # Add the target city to the possible solution
                solution = path + [neighbor]
                pathDistance = distance(solution)
                #print(solution)
                # Keep track of the smallest cost and best route
                if smallestCost == 0:
                    smallestCost = pathDistance
                    bestRoute = solution
                elif pathDistance < smallestCost:
                    smallestCost = pathDistance
                    bestRoute = solution
                #print(bestRoute)
            else:
                # Append the Stack if endCity is not met
                stack.append((neighbor, path + [neighbor]))

    return bestRoute, smallestCost, (perf_counter() - startTime)

if __name__ == '__main__':
    # Assigns the specified file to a variable
    file = r"11PointDFSBFS.tsp"
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

    # Gets the number of cities in the given file
    cityCount = len(coordinates)

    ## GENERATES OUTPUT FOR EACH INTERPRETATION. UNCOMMENT AS YOU DESIRE #####

    #### BFS OUTPUT - LEAST NODES#######
    bestRoute, smallestCost, runTime = BFS(euclidianPaths, 1, 11)
    print("\n***** BFS Algorithm: Least Nodes Route *****")
    print("Route: " + str(bestRoute))
    print("Cost: " + str(smallestCost))
    print("Runtime: " + str(runTime))

    #### DFS OUTPUT - FIRST DEEP#######
    stack = [1]
    visitedNodes = [1]
    bestRoute, smallestCost, runTime = DFS(stack)
    print("\n***** DFS Algorithm: First Deep Route to Destination*****")
    print("Route: " + str(bestRoute))
    print("Cost: " + str(smallestCost))
    print("Runtime: " + str(runTime))

    #### BFS OUTPUT - CHEAPEST #######
    bestRoute, cost, runtime = CheapestBFS(euclidianPaths, 1, 11)
    print("\n***** BFS Algorithm: Cheapest Route *****")
    print("Route: " + str(bestRoute))
    print("Cost: " + str(cost))
    print("Runtime: " + str(runtime))

    #### DFS OUTPUT - CHEAPEST #######
    bestRoute, smallestCost, runTime = CheapestDFS(euclidianPaths, 1, 11)
    print("\n***** DFS Algorithm: Cheapest Route *****")
    print("Route: " + str(bestRoute))
    print("Cost: " + str(smallestCost))
    print("Runtime: " + str(runTime))

    cities = len(bestRoute)

    #### GRAPHING of PATHS #######
    # Sets the main plot
    plt.title('BFS & DFS: Cheapest Route')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')

    # Gets the coordinates of each city in the original route
    x = []
    y = []
    for i in range(1, cityCount + 1):
        city = i
        x.append(coordinates[city][0])
        y.append(coordinates[city][1])

    # Gets the coordinates of each city of the new optimized route (bestRoute)
    ox = []
    oy = []
    for i in range(len(bestRoute)):
        city = bestRoute[i]
        ox.append(coordinates[city][0])
        oy.append(coordinates[city][1])

    # Calculates the individual path connections for graphing purposes
    individualPaths = []
    for c in euclidianPaths:
        for n in euclidianPaths[c]:
            individualPaths.append([c,n])

    #print(individualPaths)
    # Appends the coordinates of each city in the individual paths for graphing
    nx = []
    ny = []
    for i in range(len(individualPaths)):
        for j in individualPaths[i]:
            nx.append(coordinates[j][0])
            ny.append(coordinates[j][1])

    # Outputs the data collected by far and isolates the solution in a different color 
    plt.scatter(nx,ny, marker ='o')
    plt.plot(nx,ny, marker = 'o')
    for i in range(cityCount):
        city = i + 1
        plt.annotate(city, (x[i], y[i]))
    plt.scatter(ox, oy, c = 'red', marker = 'o')
    plt.plot(ox, oy, c= 'red')

    plt.show()



