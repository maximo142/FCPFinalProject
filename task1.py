import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
from math import exp
import sys


def create_array(rows, cols):
    '''
    Generate a 2D grid for the Ising model with significant size.
    Each cell in the grid is randomly set to either -1 or 1
    Returns:
        numpy.ndarray: A 2D array with randomly assigned spins of -1 or 1.
    '''
    if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:  # make sure the inputs are
        # positive intgers
        raise ValueError("Rows and columns must be positive integers.")
    return np.random.choice([-1, 1], size=(rows, cols))


a1 = create_array(100, 100)  # creat 100 by 100 array called a1


def calculate_agreement(population, row, col, H=0):
    '''
    This function should return the extent to which a cell agrees with its neighbours.
    Inputs:
        population (numpy array)
        row (int)
        col (int)
        H (float): external magnetic field acting as a bias
    Returns:
        agreement (float)
    '''
    n_rows, n_cols = population.shape  # Get the dimensions of the array

    # Indices of the direct neighbors
    neighbors_indices = [
        ((row - 1) % n_rows, col),  # Up
        (row, (col - 1) % n_cols),  # Left
        (row, (col + 1) % n_cols),  # Right
        ((row + 1) % n_rows, col)  # Down
    ]
    neighbors_values = [population[r, c] for r, c in
                        neighbors_indices]  # takeout the neighbors values and put in a list
    cell_current_value = population[row, col]

    # agreement as the sum of the products of neighboring values and the cell's value
    agreement = sum(cell_current_value * value for value in neighbors_values) + (cell_current_value * H)

    return agreement


def ising_step(population, alpha, H):
    '''
    This function will perform a single update of the Ising model
    Inputs:
        population (numpy array)
        H (float): external magnetic field
        alpha (float): inverse temperature, controls stochastic flipping
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, H)

    if agreement < 0:
        population[row, col] *= -1

    probability = math.exp(-agreement / alpha)  # use the probalitity formula from Task 1 (condition 2)

    random_float = random.random()  # generate a random float from 0 to 1 to take the correct porbability (condition 2)

    # condition 2: even when agreement is  positive, this condition we choose to accept flips that reduce agreement with probability
    if agreement > 0 and probability > random_float:
        population[row, col] *= -1


def plot_ising(im, population):
    '''
    Display a plot of the Ising model by updating the plot image with the new population state.

    Args:
    im (matplotlib.image.AxesImage): The image object returned by a previous imshow call.
    population (numpy array): The grid of the Ising model where cells are either -1 or 1.
    '''
    # Map Ising model values to color values: 255 for -1, and 0 for 1
    new_im = np.array([[0 if val == 1 else 255 for val in row] for row in population], dtype=np.uint8)
    im.set_data(new_im)
    plt.draw()
    plt.pause(0.1)  # pause to update the plot


def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''
    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert calculate_agreement(population, 1, 1) == 4, "Test 1"

    population[1, 1] = 1.
    assert calculate_agreement(population, 1, 1) == -4, "Test 2"

    population[0, 1] = 1.
    assert calculate_agreement(population, 1, 1) == -2, "Test 3"

    population[1, 0] = 1.
    assert calculate_agreement(population, 1, 1) == 0, "Test 4"

    population[2, 1] = 1.
    assert calculate_agreement(population, 1, 1) == 2, "Test 5"

    population[1, 2] = 1.
    assert calculate_agreement(population, 1, 1) == 4, "Test 6"

    # Testing external pull (H)
    population = -np.ones((3, 3))
    assert calculate_agreement(population, 1, 1, 1) == 3, "Test 7"
    assert calculate_agreement(population, 1, 1, -1) == 5, "Test 8"
    assert calculate_agreement(population, 1, 1, 10) == -6, "Test 9"
    assert calculate_agreement(population, 1, 1, -10) == 14, "Test 10"

    print("Tests passed")


def ising_main(population, alpha, H):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, alpha, H)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


def flags():
    Main = False  # boolean flag to determine whether to run ising_main
    alpha = 1  # default value for alpha
    H = 0  # default value for H (the external magnetic field)

    #  loop through arguments to set simulation parameters
    for args in range(len(sys.argv)):
        if sys.argv[args] == "-test_ising":
            test_ising()  # run the test_ising function if the flag is present
        if sys.argv[args] == "-H":
            H = float(sys.argv[args + 1])  # set H to the next argument value if the flag is present
        if sys.argv[args] == "-alpha":
            alpha = float(sys.argv[args + 1])  # set alpha to the next argument value if the flag is present
        if sys.argv[args] == "-ising_main":
            Main = True  # this flag will help to run the function with the initial values of H and alpha as asked

        # this will help to add diffrent values for H and alpha and run the function to see the change
    if Main == True:
        ising_main(population=a1, alpha=alpha, H=H)  # Ensure 'a1'(array) is defined or imported appropriately here we
        # call the population array we have created (a1)


# be sure the function only runs when the script is executed
if __name__ == "__main__":
    flags()
