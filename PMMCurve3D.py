"""
By: Karim Laknejadi, Ph.D. in Structural Engineering
This script generates a 3D interaction curve for P-M-M (P-Moment-Moment) in a defined space.
It uses matplotlib, numpy and shapely libraries to visualize the interaction between axial force (P) 
and biaxial moments (M, M). The resulting plot helps in analyzing the structural stability 
and strength of elements subjected to combined loads.
This is based on ACI318 Code.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from shapely.geometry import Polygon, box
from shapely.geometry import Polygon, LineString, Point # type: ignore
from shapely.ops import split
import plotly.graph_objects as go
import yaml
from dataclasses import dataclass

@dataclass
class sectionData:
    cornerCoordinates: list
    fc: float
    fy: float
    Es: float
    alphaSteps: float
    numberofPoints: int
    beta1: float
    includePhiFactors: bool
    cover: float
    bar_diameters: list
    bar_positions: list
    bar_areas: list
    cCorners: list

def read_yaml(file_path):
    # Load YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Extract coordinates
    # these are the location of the eplison max corners in the sectionCorners list
    [cC, cornerCoordinates, barCoordinates] = coordinate_modifier(data)
    beta1 = 0.85 if data['fc'] <= 4 else (0.85 - 0.05 * (data['fc'] - 4))  # ACI 22.2.2.4.3
    beta1 = max(beta1, 0.65)
    dataIn = sectionData(
        fc = data['fc'],
        fy = data['fy'],
        Es = data['Es'],
        alphaSteps = data['alphaSteps'],
        numberofPoints = data['numberofPoints'],
        beta1 = beta1,
        cornerCoordinates = cornerCoordinates,
        includePhiFactors = data['includePhiFactors'],
        cover = data['cover'],
        bar_diameters = data['bar_diameters'][0],
        bar_areas = data['bar_areas'][0],
        bar_positions = barCoordinates,
        cCorners = cC
    )
    return dataIn

def coordinate_modifier(data):
    # Extract coordinates, considering the coordinate system of the section is located at the (xmin, ymin) point of the section
    xmin = min([data['concreteSectionCoordinates'][k][0] for k in range(len(data['concreteSectionCoordinates']))])
    ymin = min([data['concreteSectionCoordinates'][k][1] for k in range(len(data['concreteSectionCoordinates']))])
    cornerCoordinates = [(point[0]- xmin, point[1]- ymin) for point in data['concreteSectionCoordinates']]
    barCoordinates = [(point[0]- xmin, point[1]- ymin) for point in data['bar_positions']]

    concrete_area = polygon_area(cornerCoordinates)
    concrete_centroid = polygon_centroid(cornerCoordinates, concrete_area)

    # Find the farthest points from the centroid in each quadrant
    quadrants = {
        'Q1': [],  # x >= centroid_x, y >= centroid_y
        'Q2': [],  # x < centroid_x, y >= centroid_y
        'Q3': [],  # x < centroid_x, y < centroid_y
        'Q4': []   # x >= centroid_x, y < centroid_y
    }
    
    for point in cornerCoordinates:
        x, y = point
        if x >= concrete_centroid[0] and y >= concrete_centroid[1]:
            quadrants['Q1'].append(point)
        elif x < concrete_centroid[0] and y >= concrete_centroid[1]:
            quadrants['Q2'].append(point)
        elif x < concrete_centroid[0] and y < concrete_centroid[1]:
            quadrants['Q3'].append(point)
        else:
            quadrants['Q4'].append(point)
    
    farthest_points = {}
    for quadrant, quadrant_points in quadrants.items():
        if not quadrant_points:
            farthest_points[quadrant] = None
            continue
        farthest_point = max(quadrant_points, key=lambda p: distance(p, concrete_centroid))
        farthest_points[quadrant] = farthest_point

    # the corner coordinates are the location of the eplison max corners in the sectionCorners list
    cC = []
    for corner in range(len(cornerCoordinates)):
        if cornerCoordinates[corner] == farthest_points['Q3']:
            cC.append(corner)
        elif cornerCoordinates[corner] == farthest_points['Q4']:
            cC.append(corner)
        elif cornerCoordinates[corner] == farthest_points['Q1']:
            cC.append(corner)
        elif cornerCoordinates[corner] == farthest_points['Q2']:
            cC.append(corner)

    return [cC, cornerCoordinates, barCoordinates]

def distance(point1, point2):
    """Calculate the Euclidean distance between two 2D points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def polygon_area(vertices):
    """
    Calculate the area of a polygon given its vertices using the shoelace formula.
    Parameters:
    - vertices: List of tuples representing the coordinates of the vertices [(x1, y1), (x2, y2), ...].

    Returns:
    - area: Area of the polygon (float).
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Next vertex (wraps around to the first vertex)
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0

def polygon_centroid(vertices, area):
    """
    Calculate the centroid of a polygon given its vertices.
    Parameters:
    - vertices: List of tuples representing the coordinates of the vertices [(x1, y1), (x2, y2), ...].

    Returns:
    - centroid: Centroid of the polygon (tuple of floats, e.g., (x, y)).
    """
    n = len(vertices)
    cx, cy = 0.0, 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Next vertex (wraps around to the first vertex)
        factor = (x1 * y2 - x2 * y1)
        cx += (x1 + x2) * factor
        cy += (y1 + y2) * factor
    cx /= 6 * area
    cy /= 6 * area
    return (cx, cy)

def compute_gross_centroid(concrete_vertices, rebar_areas, rebar_positions):
    """
    Compute the centroid of the gross section of a reinforced concrete section.
    Parameters:
    - concrete_vertices: List of tuples representing the coordinates of the concrete section's vertices [(x1, y1), (x2, y2), ...].
    - rebar_areas: List of areas of the rebars (list of floats).
    - rebar_positions: List of positions of the rebars (list of tuples, e.g., [(x1, y1), (x2, y2), ...]).

    Returns:
    - centroid: Centroid of the gross section (tuple of floats, e.g., (x, y)).
    """
    # Calculate the area and centroid of the concrete section
    concrete_area = polygon_area(concrete_vertices)
    concrete_centroid = polygon_centroid(concrete_vertices, concrete_area)

    # Calculate the total area of the rebars
    total_rebar_area = sum(rebar_areas)

    # Calculate the weighted sum of centroids for the rebars
    rebar_weighted_x = sum(area * pos[0] for area, pos in zip(rebar_areas, rebar_positions))
    rebar_weighted_y = sum(area * pos[1] for area, pos in zip(rebar_areas, rebar_positions))

    # Calculate the total area of the gross section
    total_area = concrete_area + total_rebar_area

    # Calculate the centroid of the gross section
    centroid_x = (concrete_area * concrete_centroid[0] + rebar_weighted_x) / total_area
    centroid_y = (concrete_area * concrete_centroid[1] + rebar_weighted_y) / total_area

    return (centroid_x, centroid_y)

def compute_area_of_split_polygon(polygon_coords, line_coords):
    # Define the polygon (e.g., a square)
    polygon = Polygon(polygon_coords)
    # Define the line (e.g., a diagonal line)
    line = LineString(line_coords)
    topParts = {'area':0, 'XC':0, 'YC': 0}
    botParts = {'area':0, 'XC':0, 'YC': 0}
    output = {'status': False,  
                'position1': '',  
                'partArea1': 0, 
                'xCentroid1': 0,
                'yCentroid1': 0,
                'position2': '',
                'partArea2': 0, 
                'xCentroid2': 0,
                'yCentroid2': 0}
    # Check if the line intersects the polygon
    if line.intersects(polygon):
        output['status'] = True
        # Split the polygon using the line
        result = split(polygon, line)
        for i, part in enumerate(result.geoms):
            part_centroid = part.centroid
            point = Point(part.centroid)
            # Find the closest point on the line to the given point
            closest_point = line.interpolate(line.project(point))
            # Determine if the area is on the top or bottom of the line
            if point.y >= closest_point.y: # top of the line
                topParts['XC'] = (topParts['XC']*topParts['area'] + part.area*part_centroid.x)/(topParts['area'] + part.area)
                topParts['YC'] = (topParts['YC']*topParts['area'] + part.area*part_centroid.y)/(topParts['area'] + part.area)
                topParts['area'] += part.area
            else: # bottom, under the line
                botParts['XC'] = (botParts['XC']*botParts['area'] + part.area*part_centroid.x)/(botParts['area'] + part.area)
                botParts['YC'] = (botParts['YC']*botParts['area'] + part.area*part_centroid.y)/(botParts['area'] + part.area)
                botParts['area'] += part.area

        output["position1"] = 'top'
        output["partArea1"] = topParts['area']
        output["xCentroid1"] = topParts['XC']
        output["yCentroid1"] = topParts['YC']
        
        output["position2"] = 'bottom'
        output["partArea2"] = botParts['area']
        output["xCentroid2"] = botParts['XC']
        output["yCentroid2"] = botParts['YC'] 
    else:
        output['status'] = False
    return output

def elementSurface():
    # Step 1: Define the arbitrary shape (e.g., a polygon)
    shape = Polygon([(0, 0), (4, 0), (3, 3), (1, 4), (0, 2)])

    # Step 2: Define the grid parameters
    grid_size = 0.05  # Size of each square in the grid

    xmin, ymin, xmax, ymax = shape.bounds  # Bounding box of the shape

    # Generate grid coordinates
    x_coords = np.arange(xmin, xmax + grid_size, grid_size)
    y_coords = np.arange(ymin, ymax + grid_size, grid_size)

    # Step 3: Create grid cells and check if they intersect with the shape
    cells = []
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            # Create a square cell
            cell = box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
            # Check if the cell intersects with the shape
            if cell.intersects(shape):
                cells.append(cell)

    # Step 4: Extract coordinates of each cell
    cell_coordinates = [list(cell.exterior.coords) for cell in cells]
    return cell_coordinates

def distFromLine(line, point): # line: [A, B, C]  point: [x0,y0]
    """
    Compute the distance of a point (x0, y0) from a line defined by Ax + By + C = 0.
    Parameters:
    A, B, C (float): Coefficients of the line equation Ax + By + C = 0.
    x0, y0 (float): Coordinates of the point.

    Returns:
    float: The distance from the point to the line.
    """
    data = {'dist':0, 'position': []}
    numerator = -line[0] * point[0] + 1 * point[1] - line[1]
    denominator = math.sqrt(line[0]**2 + 1**2)
    data['dist'] = abs(numerator)/denominator
    data['position'] = 'top' if numerator >= 0 else 'bottom'
    return data

def cornerChange(data, cline, corner, concPressure):
    posData = distFromLine(cline, corner)
    c0 = posData['dist']

    cornerChangeStatus = False
    for x,y in data.cornerCoordinates:
        posData = distFromLine(cline, [x, y])
        ci = posData['dist']
        loci = posData['position']
        if loci == concPressure and ci > c0:
            c = ci
            corner = [x, y]
            cornerChangeStatus = True
    return [cornerChangeStatus, corner]

# Stress block parameters
def stress_block_params(c):
    beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 4)))
    a = beta1 * c
    return a, beta1

def Scatter3D(data, dataIn):
    # Generate random 3D point cloud
    np.random.seed(42)
    x, y, z, epsilon , c, alphaI, scs = [], [], [], [], [], [], []
    for alpha in data.keys():
        x.extend(data[alpha][1]) # Mx
        y.extend(data[alpha][2]) # My
        z.extend(data[alpha][0]) # P   
        c.extend(data[alpha][3]) #
        epsilon.extend(data[alpha][4])
        alphaI.extend(data[alpha][5])
        scs.extend(data[alpha][6])

    # Plot
    fig = go.Figure()
    if dataIn.includePhiFactors:
        titleT = f"P-M-M Interaction Curve (ACI 318-19), Reinforced Concrete Section, Included phi factors"
    else:
        titleT = f"P-M-M Interaction Curve (ACI 318-19), Reinforced Concrete Section, Excluded phi factors"
    
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode = 'lines+markers',
        marker = dict(size = 3, color = z, colorscale='Viridis', opacity = 0.8),
        text = c,  # Text to display on hover
        customdata = list(zip(epsilon, alphaI, scs)),
        hovertemplate = 'P: %{z}<br>' +  # Rename z
                        'Mx: %{x}<br>' +  # Rename x
                        'My: %{y}<br>' +  # Rename y
                        'epsilon_t: %{customdata[0]}<br>' +  # Display additional data
                        'c: %{text}<br>' +  # Custom hover text
                        'alpha: %{customdata[1]}<br>' +  # Display additional data
                        'SecStatus: %{customdata[2]}<br>' +  # Display additional data
                        '<extra></extra>'  # Remove trace name
    ))
    fig.update_layout(
        scene=dict(
            xaxis = dict(
                title = 'Mx (kips-in)',
                showgrid=True,
                gridcolor='gray',
                gridwidth=2,
                title_font = dict(
                    family='Courier New',  # Font family
                    size=24,               # Font size
                    color='black'           # Font color
                )
            ),
            yaxis = dict(
                title = 'My (kips-in)',
                showgrid=True,
                gridcolor='gray',
                gridwidth=2,
                title_font = dict(
                    family='Courier New',  # Font family
                    size=24,               # Font size
                    color='black'           # Font color
                )
            ),
            zaxis = dict(
                title = 'P (kips)',
                showgrid=True,
                gridcolor='gray',
                gridwidth=2,
                title_font = dict(
                    family='Courier New',  # Font family
                    size=24,               # Font size
                    color='black'           # Font color
                )
            )
        ),
        title = titleT,
        title_font = dict(
            size = 32,
            family = 'Arial',
            color='black',
        ),
        hoverlabel = dict(
            font_size = 18,               # Set the font size for hover text
            font_family = 'Courier New',  # Optional: Set the font family
            font_color = 'black'          # Optional: Set the font color
        ),
    )

    fig.show()
    return 0

def Scatter2D(data, dataIn):
        
    # Plotting
    font1 = {'family': 'serif',  'color':  'darkred', 'weight': 'normal', 'size': 10}  
    font2 = {'family': 'serif',  'color':  'darkred', 'weight': 'normal', 'size': 8}  

    # Create a figure and a 2x2 grid of subplots  
    fig, axs = plt.subplots(4, 3, figsize=(12, 10))  # Adjust figsize as needed 
    alpha_list = list(data.keys())
    k = [0,0,0,0]
    for z in range(4):
        k[z] = np.argmin([np.abs(alpha_list[i] - z*np.pi/2) for i in range(len(alpha_list))])

    # Plot data on each subplot
    for i in range(4):
        if i == 0 or i == 2:
            axs[i, 0].plot(data[alpha_list[k[i]]][2], data[alpha_list[k[i]]][0])  
            axs[i, 0].set_title(f"P vs My, teta = {np.round(alpha_list[k[i]]*180/np.pi,1)} (Deg)", fontdict = font1)
            axs[i, 0].set_xlabel('Design Moment - My (kips-in)', fontdict = font2)
            axs[i, 0].set_ylabel('Axial Load (kips)', fontdict = font2)
        else:
            axs[i, 0].plot(data[alpha_list[k[i]]][1], data[alpha_list[k[i]]][0])
            axs[i, 0].set_title(f"P vs Mx, teta = {np.round(alpha_list[k[i]]*180/np.pi,1)} (Deg)", fontdict = font1)
            axs[i, 0].set_xlabel('Design Moment - Mx (kips-in)', fontdict = font2)
            axs[i, 0].set_ylabel('Axial Load (kips)', fontdict = font2)

        axs[i, 1].plot(data[alpha_list[k[i]]][3], data[alpha_list[k[i]]][0])  
        axs[i, 1].set_title(f"P vs c, teta = {np.round(alpha_list[k[i]]*180/np.pi,1)} (Deg)", fontdict = font1)
        axs[i, 1].set_xlabel('Neutral Axis Depth - c (in)', fontdict = font2)
        axs[i, 1].set_ylabel('Axial Load (kips)', fontdict = font2)

        axs[i, 2].plot(data[alpha_list[k[i]]][4], data[alpha_list[k[i]]][0])
        axs[i, 2].set_title(f"P vs epsilon-t, teta = {np.round(alpha_list[k[i]]*180/np.pi,1)} (Deg)", fontdict = font1)
        axs[i, 2].set_xlabel('Maximum Tensile Strain - epsilon-t', fontdict = font2)
        axs[i, 2].set_ylabel('Axial Load (kips)', fontdict = font2)        

    # Add a general title to the figure
    if dataIn.includePhiFactors:
        titleT = f"P-M-M Interaction Curve (ACI 318-19), Reinforced Concrete Section, Included phi factors"
    else:
        titleT = f"P-M-M Interaction Curve (ACI 318-19), Reinforced Concrete Section, Excluded phi factors"
    fig.suptitle(titleT, fontsize = 16)  
    # Adjust layout to prevent overlapping titles/labels  
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  

    # Draw a custom line separating subplots

    line = mlines.Line2D([0, 1], [0.255, 0.255], color='gray', linestyle='--', linewidth = 2)
    fig.add_artist(line)
    line = mlines.Line2D([0, 1], [0.475, 0.475], color='gray', linestyle='--', linewidth = 2)
    fig.add_artist(line)
    line = mlines.Line2D([0, 1], [0.69, 0.69], color='gray', linestyle='--', linewidth = 2)
    fig.add_artist(line)

    # Show the plot  
    plt.show() 
    return 0

# Generate P-M-M interaction curve
def calculate_pmm_interaction(data):
    # Iterate over neutral axis depths (c)
    c_min = 0.001
    c_max = h * 10  # Sufficiently large to cover all cases
    num_points = data.numberofPoints
    c_list = np.linspace(c_min, c_max, num_points)
    alpha_list = np.linspace(0, 8*np.pi/4, int(360/data.alphaSteps))  # Step size for Angle of neutral axis (radians)
    dataDict = {}

    for alpha in alpha_list:
        alpha = 0.98*alpha if abs(alpha - np.pi/2)<0.001 or abs(alpha - 3*np.pi/2)<0.001 else alpha 
        P_alpha_list, Mx_alpha_list, My_alpha_list, epsilon_t_list, cs_list, alpha_list_all , sectionStatus = [], [], [], [], [], [], []
        # initial guess of the corner of the section with maximum comprssive strain ( = 0.003)
        if alpha < np.pi/2:                        # top left corner
            corner = data.cornerCoordinates[data.cCorners[3]]
        elif alpha >= np.pi/2 and alpha < np.pi:   # bottom left corner
            corner = data.cornerCoordinates[data.cCorners[0]] 
        elif alpha >= np.pi and alpha < 3*np.pi/2: # bottom right corner
            corner = data.cornerCoordinates[data.cCorners[1]]
        elif alpha >= 3*np.pi/2:                   # top right corner
            corner = data.cornerCoordinates[data.cCorners[2]]

        runStatus = True
        mi = np.tan(alpha)
        while runStatus:
            # Initialize lists for interaction points
            P_list, Mx_list, My_list, ci_list, epsiloni_t_list, scS = [], [], [] , [], [], []
            for c in c_list:
                a = data.beta1 * c
                y0a = corner[1] - a/(np.cos(alpha)) - corner[0]*np.tan(alpha)
                y0c = corner[1] - c/(np.cos(alpha)) - corner[0]*np.tan(alpha)
                cline = [mi, y0c] #  (m, c) for the line of the neutral axis

                lineNodes = [[-b, -b*mi + y0a], [2*b, 2*b*mi + y0a]]
                areaData = compute_area_of_split_polygon(data.cornerCoordinates, lineNodes)
                
                # the location of the compression area of concrete  with respect to the neutral axis
                if alpha <= np.pi/2: 
                    concPressure = 'top'
                elif alpha > np.pi/2 and alpha < 3*np.pi/2:
                    concPressure = 'bottom'
                else:
                    concPressure = 'top'
                
                if areaData['status']: # the considered neutral axis is cutting the section
                    compArea = areaData['partArea1'] if areaData['position1'] == concPressure else areaData['partArea2']
                    # controid of the compression area
                    Xc = areaData['xCentroid1'] if areaData['position1'] == concPressure else areaData['xCentroid2']
                    Yc = areaData['yCentroid1'] if areaData['position1'] == concPressure else areaData['yCentroid2']
                else: # the considered neutral axis is NOT cutting the section
                    compArea = Ag
                    Xc = centroid[0]
                    Yc = centroid[1]
                
                # check if the corner (the point with maximum compressive strain) has changed
                if runStatus:
                    [cornerChangeStatus, corner] = cornerChange(data, cline, corner, concPressure)
                
                if cornerChangeStatus:
                    runStatus = True
                else:
                    # Steel contribution
                    runStatus = False
                    P_steel = 0.0
                    Mx_steel = 0.0
                    My_steel = 0.0
                    epsilon_i_y_list = []
                    for pi, Asi in zip(data.bar_positions, data.bar_areas):
                        xi = pi[0]
                        yi = pi[1]
                        posData = distFromLine(cline, [xi, yi])
                        ci = posData['dist']
                        loci = posData['position']
                        epsilon_i_y = 0.003 * ci / c if loci == concPressure else -0.003 * ci / c
                        compArea = compArea - Asi if loci == concPressure else compArea

                        stress_i_y = data.Es * epsilon_i_y if abs(epsilon_i_y) < epsilon_y else data.fy*np.sign(epsilon_i_y)
                        Fsi = Asi * stress_i_y

                        P_steel += Fsi
                        Mx_steel += Fsi * (xi - centroid[0])
                        My_steel += Fsi * (yi - centroid[1])
                        epsilon_i_y_list.append(epsilon_i_y)

                    # Concrete contribution
                    concrete_force = 0.85 * data.fc * compArea
                    # here y_bar should be calculated based on the position of the compression area
                    concrete_moment_x = concrete_force * (Xc - centroid[0])
                    concrete_moment_y = concrete_force * (Yc - centroid[1])

                    # Nominal strengths
                    Pn = concrete_force + P_steel
                    Mnx = concrete_moment_x + Mx_steel
                    Mny = concrete_moment_y + My_steel

                    # Cap Pn at Pn_max_ACI
                    Pn_max_ACI = 0.80 * (0.85 * data.fc * (Ag - Ast) + data.fy * Ast)
                    if Pn > Pn_max_ACI:
                        Pn = Pn_max_ACI
                        Mnx = 0.0
                        Mny = 0.0

                    # Strain in extreme tension steel
                    epsilon_t_max  = min(epsilon_i_y_list)

                    # Strength reduction factor (phi)
                    if data.includePhiFactors:
                        if epsilon_t_max > -epsilon_y: # Compression-Controlled Section
                            # the strain in the extreme tension steel, has a strain that is compressive
                            phi = 0.65
                        elif epsilon_t_max < -epsilon_y and epsilon_t_max > -(epsilon_y + 0.003): # Compression-Controlled Section
                            phi = 0.65 + (-epsilon_t_max - 0.002) * (0.25 / 0.003)
                        elif epsilon_t_max < -(epsilon_y + 0.003):
                            phi = 0.9
                        else:
                            phi = 0.65
                    else:
                        phi = 1.0

                    # Design strengths
                    if epsilon_t_max >= -epsilon_y:    # Compression-controlled section
                        sci = 'CC'
                    elif epsilon_t_max <= -epsilon_y: # Tension-controlled section
                        sci = 'TC'
                    else:                             # Transition Zone
                        sci = 'TZ'

                    design_P = phi * Pn
                    design_Mx = phi * Mnx
                    design_My = phi * Mny

                    P_list.append(np.round(design_P,2))
                    Mx_list.append(np.round(design_Mx,2))
                    My_list.append(np.round(design_My,2))
                    ci_list.append(np.round(c,2))
                    epsiloni_t_list.append(np.round(min(epsilon_t_max, -2*0.005), 5))
                    scS.append(sci)

        P_alpha_list.extend(P_list)
        Mx_alpha_list.extend(Mx_list)
        My_alpha_list.extend(My_list)
        cs_list.extend(ci_list)
        epsilon_t_list.extend(epsiloni_t_list)
        alpha_list_all.extend([np.round(alpha,3) for k in range(len(ci_list))])
        sectionStatus.extend(scS)


        dataDict[alpha] = [P_alpha_list, 
                           Mx_alpha_list, 
                           My_alpha_list, 
                           cs_list, 
                           epsilon_t_list, 
                           alpha_list_all, 
                           sectionStatus]

    return dataDict

if __name__ == "__main__":
    # Input parameters
    data = read_yaml('sectionData.yaml')

    b = max(data.cornerCoordinates[k][0] for k in range(len(data.cCorners))) - min(data.cornerCoordinates[k][0] for k in range(len(data.cCorners)))   # Column width (inches)
    h = max(data.cornerCoordinates[k][1] for k in range(len(data.cCorners))) - min(data.cornerCoordinates[k][1] for k in range(len(data.cCorners)))  # Column depth (inches)

    xOrigin = min([data.cornerCoordinates[k][0] for k in range(len(data.cornerCoordinates))])
    yOrigin = min([data.cornerCoordinates[k][1] for k in range(len(data.cornerCoordinates))])

    # Calculate the gross centroid of the section
    centroid = compute_gross_centroid(data.cornerCoordinates, data.bar_areas, data.bar_positions)

    # Derived parameters
    Ag = polygon_area(data.cornerCoordinates) # Gross area of the section (in²)
    Ast = sum(data.bar_areas)  # Total steel area (in²)
    epsilon_y = data.fy / data.Es  # Yield strain

    # Generate P-M-M interaction curve
    dataDict = calculate_pmm_interaction(data)
    P_list = []
    for key in dataDict.keys():
        P_list.extend(dataDict[key][0])
    print(f"Pmin = {min(P_list)}, Pmax = {max(P_list)}")

    Scatter3D(dataDict, data)
    Scatter2D(dataDict, data)

