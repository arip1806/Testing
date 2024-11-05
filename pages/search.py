import matplotlib.pyplot as plt

# Get user input for city names and coordinates
num_cities = int(input("Enter the number of cities: "))
city_names = []
city_coords = []

for i in range(num_cities):
    city_name = input(f"Enter the name of city {i+1}: ")
    x_coord = float(input(f"Enter the x-coordinate for {city_name}: "))
    y_coord = float(input(f"Enter the y-coordinate for {city_name}: "))
    city_names.append(city_name)
    city_coords.append((x_coord, y_coord))

# Extract x and y coordinates
x_coords = [coord[0] for coord in city_coords]
y_coords = [coord[1] for coord in city_coords]

# Create a scatter plot
plt.scatter(x_coords, y_coords)

# Add labels for each city
for i, city in enumerate(city_names):
    plt.annotate(city, (x_coords[i], y_coords[i]))

plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("City Coordinates")
plt.grid(True)
plt.show()
