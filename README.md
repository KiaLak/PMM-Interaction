# 3D P-M-M-Interaction / Interaction Diagram for Reinforced Concrete Columns
This python code generates a 3D P-M-M interaction curve for a reinforced concrete section with an arbitrary geometric shape and reinforcement layout, incorporating the requirements of ACI 318-19.

The 3D P-M-M interaction diagram for reinforced concrete sections is a critical tool in structural engineering for analyzing and designing columns subjected to combined axial force and biaxial bending moments, as per ACI 318-19. This three-dimensional diagram represents the interaction between axial load (P) and bending moments about two orthogonal axes (Mx and My), providing a comprehensive visualization of the section's capacity under various loading conditions. The diagram is constructed by plotting the failure envelope, which defines the limits of the section's strength based on the principles of equilibrium, compatibility, and the constitutive relationships of concrete and steel reinforcement. Engineers use this diagram to ensure that the applied loads fall within the safe zone, thereby guaranteeing the structural integrity and safety of reinforced concrete columns under complex loading scenarios.

## How to use
In the side YAML file, specify the material properties, the locations of the corners of the considered section, and the positions of all the reinforcing bars. Then, run the code. Ensure that you are using Python 3.12 or a newer version.

## Output
- 3D Interaction P-M-M Diagram, On each point on this diagram, all information about that point are shown including:

    1- P (Axial Force)
  
    2- Mx (Moment in x dir.)
  
    3- My (Moment in y dir.)
    4- Maximum tensile strain in the bars
    5- c, the location of the neutral axis
    6- Alpha (rad), the location of the considered surface
    7- The status of the section (Compression-controlled, Tension-controlled, Transition Zone)
- 2D Interaction P-M Digrams for surfaces. with alpha equal to 0, 90, 180, and 270 degrees.

![image](https://github.com/user-attachments/assets/505bce60-5b32-49e9-8d23-2e41a319ca10)





