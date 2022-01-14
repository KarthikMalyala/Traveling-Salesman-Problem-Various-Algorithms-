# Karthik Malyala
# Project 1 (TSP - Brute Force)
# CSE 545 - Artificial Intelligence

# Imports the required libraries
from itertools import permutations
import matplotlib.pyplot as plt
import time

# Assigns the specified file to a variable
file = "C:\Fall 2021\CSE 545\Project 1\Random4.tsp"

coordinates = {} # Stores the cities and their corresponding coordinates in a dictionary

# Opens the file and reads line-by-line after 'NODE_COORD_SECTION' to get coordinates for each city
with open(file) as inp:
    for line in inp:
        if 'NODE_COORD_SECTION' in line:
            for line in inp:
                city = line.rstrip('\n')
                coord = city.split(' ')
                index = int(coord[0]) # Finds the City
                x = float(coord[1]) # Finds its X-Coordinate
                y = float(coord[2]) # Finds its Y-Coordinate
                coordinates[index] = (x, y) # Assign each city with a tuple of its coordinates in the dictionary
        # If we reached the end of the file then break out of the loop
        elif 'EOF' in line:
            break

# Gets the number of cities in the given file
cityCount = len(coordinates)

# Starts the timer for the Brute Force Algorithm to measure runtime at the end
startTime = time.time()

# Generates all possible permutations using the itertools library
permutations = permutations(range(1, (cityCount + 1)))

smallestCost = 0 # Initial smallestCost
bestRoute = [] # Empty bestRoute that will be overwritten with best route
count = 0 # Counter for each permutation

# Iterates through each permutation at a time to avoid MemoryError for datasets larger than 10 cities
for path in permutations:
    # Adds the starting destination to the end
    path = path + (path[0],)
    totalDistance = 0 # Sets the totalDistance for each route to 0 for every permutation (every route)
    for y in range (0, cityCount):
           city = path[y] # Takes the current city
           nextCity = path[y + 1] # Takes the next city
           # Assigns the x and y coordinates of both cities for distance calculation
           ax = coordinates[city][0]
           ay = coordinates[city][1]
           bx = coordinates[nextCity][0]
           by = coordinates[nextCity][1]

           # Utilizes the Distance Formula to calculate the distance between the two cities
           pathDistance = (((bx - ax) ** 2) + ((by - ay) ** 2)) ** 0.5
           # The totalDistance is incremented by each path's distance to find the totalDistance of the route in the end
           totalDistance += pathDistance

    # At the first iteration, the smallestCost needs to be the total distance of the first route because it is initially 0
    if (count == 0):
        smallestCost = totalDistance
        bestRoute = path
    # Compares the smallestCost by far to the current route's total distance to see if the smallestCost needs to be overwritten
    elif (totalDistance < smallestCost):
        smallestCost = totalDistance
        bestRoute = path
    # Increments the counter
    count += 1

# Prints the results calculated above
print('Shortest Path =', bestRoute,
      '\nTotal Cost =', smallestCost,
      '\nBrute Force Algorithm Runtime =', (time.time() - startTime), 'seconds')


#### GRAPHING of PATHS #######
# Sets the main plot for two graphs
graph = plt.figure(figsize=(100,100))
# Sets a subplot to the left for the original route
input = graph.add_subplot(1,2,1)
# Sets a subplot to the right for the optimized route
output = graph.add_subplot(1,2,2)

# Sets titles and labels for the plots and corresponding axes
input.set_title('Original Route', weight = 'bold')
output.set_title('Optimized Route', weight = 'bold')
input.set_xlabel('X-Coordinate')
input.set_ylabel('Y-Coordinate')
output.set_xlabel('X-Coordinate')
output.set_ylabel('Y-Coordinate')

# Initally empty lists for coordinates to be added for every city
x = []
y = []
ox = []
oy = []

# Gets the coordinates of each city in the original route
for i in range(1, cityCount + 1):
    city = i
    x.append(coordinates[city][0])
    y.append(coordinates[city][1])

# Adds the starting city's coordinates to complete the original route
x.append(x[0])
y.append(y[0])

# Gets the coordinates of each city of the new optimized route (bestRoute)
for i in range(len(bestRoute)):
    city = bestRoute[i]
    ox.append(coordinates[city][0])
    oy.append(coordinates[city][1])

# Plots each city according to its coordinates and connects cities with arrows, according to the order of the original route
for i in range(0, cityCount):
    input.scatter(x[i], y[i], c = 'red', marker = 'o', s = 40) # Plots city coordinates with a circle marker and red color
    cityLbl = "City "
    # Finds what city the coordinates plotted above correspond to
    for city, coords in coordinates.items():
        if coords == (x[i], y[i]):
            cityLbl += str(city)
    input.annotate(cityLbl, (x[i], y[i]), weight ='bold') # Annotates each point with a label of what city the point is representing
    input.scatter(x[i+1], y[i+1], c = 'blue', marker = 'o', s = 40) # Labels the starting city with a blue circle marker
    # Connects every city with an arrow according to the order of the original route
    input.quiver(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), angles = 'xy', scale_units = 'xy', scale = 1)

# Plots each city according to its coordinates and connects cities with arrows, according to the order of the new optimized route
for i in range(0, cityCount):
    output.scatter(ox[i], oy[i], c = 'green', marker = 'o', s = 40) # Plots city coordinates with a circle marker and red color
    cityLbl = "City "
    # Finds what city the coordinates plotted above correspond to
    for city, coords in coordinates.items():
        if coords == (ox[i], oy[i]):
            cityLbl += str(city)
    output.annotate(cityLbl, (ox[i], oy[i]), weight ='bold') # Annotates each point with a label of what city the point is representing
    output.scatter(ox[i+1], oy[i+1], c = 'blue', marker = 'o', s = 40) # Labels the starting city with a blue circle marker
    # Connects every city with an arrow according to the order of the new optimized route
    output.quiver(ox[i], oy[i], (ox[i+1] - ox[i]), (oy[i+1] - oy[i]), angles = 'xy', scale_units = 'xy', scale = 1)
# Outputs the two plots for a side-by-side comparison
plt.show()


