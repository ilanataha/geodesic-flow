# Geodesic Flow
Using different mathematical techniques to derive geodesic curves on 3D embedded surfaces, and specifically on the torus. 

## Background
Geodesic curves are in some sense curves representing the shortest path between points on a surface, or the path requiring minimal energy that ties two points.The definition of this object as a minimizer of sorts, requires defining a way to translate the local configuration of a surface around each point to another, to understand how we might follow this translation in the most economic way. This translational relation is termed in mathematics a connection. 

Geodesic flow is movement along a geodesic curve. This concept is powerful beyond the scope of purely mathematical appreciation, because it gives us a terminology with which to imagine the evolution or development of any process, moving from one point to another while its configuration space changes fundamentally, but doing so in the most conservational fashion. 

This idea motivated me to try and create visual representations for geodesic flow. In the video I created, I tried to explore and demonstrate occurences of geodesic flow in biological and physical phenomena. I was also motivated to explore the concept of a connection and the tension it imposes on singularity and objectivity, reminding us of the limited locality of our view and the multiplicity interacting and connecting each singular object or point. These scripts were written for Python script embedding in Blender software, and some of the renders made it into the video which can be viewed [here](https://vimeo.com/743570898).

## Scripting
One script uses Euler's method to approximate the curves from the geodesic equation. This has the limitation of requiring explicit equations to be inserted into the script for each surface - in this case, the equations match the torus' geodesic equation. 

Another script approaches the optimization problem enclosed in the geodesic equation directly. We look for a "minimal" path on a discretization of the given surface, using Dijkstra's Algorithm. At each vertex, the algorithm looks for the neighbor vertex minimzing not the Euclidean distance, but rather the distance between vectors being parallel transported between the respective tangent spaces.






