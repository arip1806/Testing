import streamlit as st

st.title("City Coordinates Input")
st.subheader("Enter up to 10 cities with their coordinates (x, y) in range 1-10.")

# Initialize lists for city names, x coordinates, and y coordinates
city_names = []
x_coords = []
y_coords = []

# Input fields for each city
for i in range(1, 11):
    st.write(f"City {i}")
    city_name = st.text_input(f"City {i} name", key=f"city_name_{i}")
    x_coord = st.number_input(f"x-coordinate (City {i})", min_value=1.0, max_value=10.0, step=0.1, key=f"x_coord_{i}")
    y_coord = st.number_input(f"y-coordinate (City {i})", min_value=1.0, max_value=10.0, step=0.1, key=f"y_coord_{i}")

    # Append inputs to lists
    if city_name:
        city_names.append(city_name)
        x_coords.append(x_coord)
        y_coords.append(y_coord)

# Submit button
if st.button("Submit"):
    # Display entered data
    st.write("City Names:", city_names)
    st.write("X Coordinates:", x_coords)
    st.write("Y Coordinates:", y_coords)
    
    # Example dictionary for further processing
    city_coords = dict(zip(city_names, zip(x_coords, y_coords)))
    st.write("City Coordinates Dictionary:", city_coords)
