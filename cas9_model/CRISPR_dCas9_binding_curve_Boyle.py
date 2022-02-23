import numpy as np
from scipy import linalg
import sys
#PATH_HPC05 = '/home/mklein1/Energy_Landscape_dCas9/'
#sys.path.append(PATH_HPC05)
#sys.path.append('../code_general/')
from cas9_model.read_model_ID import unpack_parameters


''''
Type some title here later to tell yourself what the content of this particular file is.

Behrouz Eslami & Misha Klein    Depken lab
'''

'''
Main functions
'''

def calc_Chi_square(parameters,xdata,ydata,yerr,
                    ONtarget_occupancy,
                    weights=np.array([1.0,1.0,1.0]),
                    guide_length=20, model_id='general_energies'):
    '''
    *** To make the Simmulated Annealing general I added the 'yerr' as argument,
    but it is really not used in our case ******************

    ******** Function coupled to Simmulated Annealing module *************
    Determine weighted chi-squared for global fit to all three datasets (if available) of Boyle et al.
    To optimize effeciency we input the mismatch configuration and calculate all quataties that are available for it.

    Also, the occupancy of the on-target is calculated before to avoid redoing this calculation for all mismatch configurations

    :param parameters:
    :param mismatch_positions:
    :param Occ_data:
    :param Asso_data:
    :param Disso_data:
    :param ONtarget_occupancy:
    :param weights:
    :param guide_length:
    :param model_id:
    :return:
    '''

    # 0) Unpack xdata, ydata and yerr from Simulated Annealing
    Occ_data = ydata[0]
    Asso_data= ydata[1]
    Disso_data=ydata[2]

    Occ_error = yerr[0]
    Asso_error= yerr[1]
    Disso_error=yerr[2]



    mismatch_positions = xdata

    # 1) For all configurations of two mismatches we have the relative bound fraction (occupancy)
    # However, we want to have the option to not fit to the occupancy map. In this case, empty data is provided
    CalculateOccupancy = False
    if len(Occ_data) > 0:
        CalculateOccupancy = True

    # 2) If we have the effective association rate predict it with the model:
    CalculateAssociation = False
    if len(Asso_data) > 0:
        CalculateAssociation = True

    # 3) If we have the effective dissociation rate predict it with the model:
    CalculateDissociation = False
    if len(Disso_data) > 0:
        CalculateDissociation = True

    # 4) Perform main calculations:
    AbsOcc_model, Asso_model, Disso_model = calc_Boyle(CalcOccupancy=CalculateOccupancy,
                                                   CalcOffRate=CalculateDissociation,
                                                   CalcOnRate=CalculateAssociation,
                                                   parameters=parameters,
                                                   mismatch_positions=mismatch_positions,
                                                   guide_length=guide_length,
                                                   model_id = model_id
                                                       )

    # 5) square difference between data and model (check which ones we have available):
    #**********************************************************************************************
    # In Boyle et al. they divide the fluoresence by on-target after 12 hrs prior to
    # fitting both asso. and disso rates. If we want to make this (minor) adjustment
    # (on target not close to saturation)
    # we just divide Asso_model and Disso_model by the ONtarget_occupancy
    #************************************************************************************************
    residual = np.zeros(3)
    if not np.math.isnan(AbsOcc_model):
        Occ_model = AbsOcc_model / ONtarget_occupancy
        residual[0] = np.sum(((Occ_data-Occ_model)/Occ_error)**2  )
    if not np.math.isnan(Asso_model):
        residual[1] = np.sum(((Asso_data - Asso_model)/Asso_error) ** 2)
    if not np.math.isnan(Disso_model):
        residual[2] = np.sum( ((Disso_data - Disso_model)/Disso_error) ** 2)




    # 6) Take weights of different maps into account (and take the sum/ inner product )
    Chi_square = weights.dot(residual)
    return Chi_square


def calc_Boyle(CalcOccupancy, CalcOffRate, CalcOnRate,
               parameters, mismatch_positions, guide_length=20, model_id='general_energies'):
    '''
    :param CalcOccupancy: boolian True if you want to calculate occupancy after 12 hrs
    :param CalcOffRate: boolian, True if you want to calculate dissocation rate
    :param CalcOnRate:  boolian, True if you want to calculate association rate
    :return:
    '''
    # 0) Initialize output (in case I do not want to calculate all 3:
    bound_fraction = float('nan')
    dissociation_rate = float('nan')
    association_rate = float('nan')

    # 1) Unpack parameters and build rate matrix to be used in all types of calculations
    rate_matrix = get_master_equation(parameters, mismatch_positions, model_id, guide_length)

    #2) Do I want (need) the occupancy after 12 hours?
    if CalcOccupancy or CalcOffRate:
        everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))
        Probability = get_Probability(rate_matrix=rate_matrix,initial_condition=everything_unbound,T=12*3600)
        bound_fraction = np.sum(Probability[1:])

    #3) Do I want the dissociation rate ?
    if CalcOffRate:
        dissociation_rate = calc_dissociation_rate(rate_matrix=rate_matrix,
                                                 initial_condition=Probability,timepoints=[500.,1000.,1500.])
    #4) Do I want the association rate?
    if CalcOnRate:
        association_rate = calc_association_rate(rate_matrix=rate_matrix,timepoints=[500.,1000.,1500.],
                                                 guide_length=guide_length)

    return bound_fraction, association_rate, dissociation_rate



'''
Helper functions
'''

def get_master_equation(parameters, mismatch_positions, model_id, guide_length):
    '''
    Construct rate matrix from given parameter set
    :param parameters:
    :param mismatch_positions:
    :param guide_length:
    :return:
    '''
    epsilon, forward_rates = unpack_parameters(parameters, model_id, guide_length)
    energies = get_energies(epsilon,mismatch_positions, guide_length)
    backward_rates = get_backward_rates(energies, forward_rates,guide_length )
    rate_matrix = build_rate_matrix(forward_rates, backward_rates)
    return rate_matrix

def get_energies(epsilon,mismatch_positions, guide_length=20):
    '''
    For general (position dependent) model make a list with the energies at every bound state
    At positions with a mismatch incorporated: add mismatch penalty (epsI[mm_pos])

    So this function returns the minima in the energy lanscape (the actual energy at every state)

    :param epsilon: [epsPAM, epsC[state_1],....,epsC[state_N],epsI[state_1],...epsI[state_N] ]
    provide as np.array()
    :param mismatch_positions: each mismach position has a range [1,2, ... , 20]
    :return: vector with the minima in the landscape
    '''
    if type(mismatch_positions)==type([]):
        mismatch_positions = np.array(mismatch_positions)
    new_epsilon = epsilon.copy()
    epsI = new_epsilon[(guide_length+1):]
    energies = -1*new_epsilon[0:(guide_length+1)] # convention: epsC>0 means downward slope
    energies[0] = new_epsilon[0]                 # convention: epsPAM>0 means upward slope
    if len(mismatch_positions)>0:
        energies[mismatch_positions.astype(int)] += epsI[(mismatch_positions.astype(int)-1)]
    return energies


def get_backward_rates(energies, forward_rates,guide_length=20):
    '''
    Apply detailed balance condition to get the backward rates from the energies and forward rates

    :param energies:
    :param forward_rates:
    :param guide_length:
    :return:
    '''
    # 0) Construct array containing backward rates
    backward_rates = np.zeros(guide_length+2)

    # 1) Apply detailed balance condition:
    backward_rates[1:] = forward_rates[:-1] * np.exp(energies)

    # 2) No rate backward from solution state
    backward_rates[0] = 0.0
    return backward_rates


def build_rate_matrix(forward_rates, backward_rates):
    '''
    build matrix in Master Equation

    :param forward_rates:
    :param backward_rates:
    :return:
    '''
    diagonal1 = -(forward_rates + backward_rates)
    diagonal2 = backward_rates[1:]
    diagonal3 = forward_rates[:-1]
    # rate_matrix = np.zeros((len(forward_rates), len(forward_rates)))  # Build the matrix

    rate_matrix = np.diag(diagonal1, k=0) + np.diag(diagonal2, k=1) + np.diag(diagonal3, k=-1)

    # for diag in range(3):
    #     if diag == 0:
    #         rate_matrix = rate_matrix + np.diag(diagonal1, k=0)
    #     elif diag == 1:
    #         rate_matrix = rate_matrix + np.diag(diagonal2, k=1)
    #     elif diag == 2:
    #         rate_matrix = rate_matrix + np.diag(diagonal3, k=-1)

    return rate_matrix

def get_Probability(rate_matrix, initial_condition,T=12*3600):
    '''
    solves the Master Equation for a given initial condition and desired time point
    :param rate_matrix: matrix with rates that makes up the Master Equation
    :param initial_condition: vector with initial configuration
    :param T: Evaluate solution at time T
    :return:
    '''
    P0 = initial_condition
    M = rate_matrix
    matrix_exponent = linalg.expm(+M*T)
    return matrix_exponent.dot(P0)

def calc_association_rate(rate_matrix,timepoints=[500.,1000.,1500.],guide_length=20,rel_concentration=0.1):
    '''
    Association experiment for the effective on-rate:
    (repeating the protocol as used in Boyle et al.)
    :return:
    '''

    #1) Calculate bound fraction for specified time points:
    # Association experiment starts with all dCas9 being unbound:
    everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))

    #3) Association rate is taken at 1nM the other data at 10nM --> adjust k_on appropriatly:
    new_rate_matrix = rate_matrix.copy()
    new_rate_matrix[0][0] *= rel_concentration  #rel_concentration=1 corresponds to 10nM
    new_rate_matrix[1][0] *= rel_concentration

    bound_fraction = []
    for time in timepoints:
        Probabilities = get_Probability(rate_matrix=new_rate_matrix,initial_condition=everything_unbound,T=time)
        bound_fraction.append( np.sum(Probabilities[1:]))

    #2) Use least-squares to fit straight line with origin forced through zero:
    kon_lin_fit = least_squares_line_through_origin(x_points=np.array(timepoints),y_points=np.array(bound_fraction))
    return kon_lin_fit

def calc_dissociation_rate(rate_matrix,initial_condition,
                           timepoints=[500.,1000.,1500.]):
    '''
    Dissociation experiment for (effective off-rate):
    method as used in Boyle et al.

    within calc_Boyle() (main function call) you will all ready calculate the occupation before flushing out dCas9
    :return:
    '''

    # 1) Flush out all freely floating dCas9 --> Set occupation in solution to zero:
    after_12hrs = initial_condition.copy()
    after_12hrs[0] = 0.0

    #2) Renormalize remainder:
    after_12hrs = after_12hrs / np.sum(after_12hrs)

    #3) Flush out all freely floating dCas9 --> Set on-rate to zero and rebuild matrix:
    new_rate_matrix = rate_matrix.copy()
    new_rate_matrix[0][0] = 0.0
    new_rate_matrix[1][0] = 0.0

    #4)  For time_k in timepoints solve the Master equation and track the fraction of dCas9 molecules in solution:
    unbound_fraction = []
    for time in timepoints:
        Probabilities = get_Probability(rate_matrix=new_rate_matrix, initial_condition=after_12hrs, T=time)
        unbound_fraction.append(Probabilities[0])

    # 5) Use least-squares to fit :
    _ , k_off_fit = least_squares(x_points=timepoints, y_points=unbound_fraction)
    return k_off_fit



def least_squares(x_points, y_points):
    size = len(x_points)
    X = np.ones((size, 2))
    X[:, 1] = x_points

    XT = X.transpose()
    Y = y_points

    a = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), Y)

    intercept = a[0]
    slope = a[1]
    return intercept, slope

def least_squares_line_through_origin(x_points, y_points):
    return np.sum( x_points*y_points )/np.sum( x_points*x_points )