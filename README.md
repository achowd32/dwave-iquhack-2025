# QAP
## Time-Dependent Quadratic Assignment Problems with Quantum Annealing
**Created by Team Qunonians for the D-Wave Challenge @ iQuHack 2025**
<br/>
Authors: Atharv Chowdhary, Rohan Kher, Leopold Li, Simon Nirenberg, Jerry Sun
<br/><br/>
We take on D-Wave's harder challenge of implementing a solution to an advanced version of the Quadratic Assignment Problem (QAP). We introduce a method for combining matrices and penalizing select cells, which allows us to solve QAP in time dependent systems, evaluate costs involved in transitioning between states, and distinguish between various kinds of actors involved in the problem formulation. We apply this method on D-Wave's Quantum Annealing systems.
## Deliverables
We offer two key deliverables as part of our submission: a Jupyter notebook which walks the user through our methodology, and a Python library which serves as a helpful abstraction for solving similar QAP problems with D-Wave's Quantum Computers.
### Jupyter Notebook
(insert)
### Python Library
Our second key deliverable for this project is the library ```ocean_qap```, which primarily builds off code from the ```qap.py``` file in this repository. A link to the complete repository for ```ocean_qap``` can be found [here](https://github.com/achowd32/ocean_qab), and the library can be installed via [this](https://pypi.org/project/ocean-qap/) link or using the command ```pip install ocean-qap```.
<br/><br/>
```ocean_qap``` provides a helpful abstraction via the ```QAP``` class, which conveniently stores and handles all the required information required for solving Quadratic Assignment Problems of the nature described by the challenge. Users can easily initialize the class, obtain optimizations, and evolve the system to obtain new optimizations based on new parameters and transition states. Various customization and visualization options are also available.
