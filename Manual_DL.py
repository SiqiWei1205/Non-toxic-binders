#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' pip install scikit-optimize')

# install packages
import tensorboard as TensorBoard
from keras.callbacks import TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args
import datetime
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# Read data
df=pd.read_csv('data_score.csv')
df

# Convert SMILES strings to RDKit molecules and visualize some of them

mols =[]
for smile in df["smiles"]:
    mols.append(Chem.MolFromSmiles(smile))
mols=[Chem.MolFromSmiles(smile) for smile in df["smiles"]]
Draw.MolsToGridImage(mols[1:10],molsPerRow=5,subImgSize=(200,200))

# Generate fingerprints using RDKit
fingerprints =[] 
safe = []
for mol_idx, mol in enumerate(mols):
    try:
        fingerprint=[x for x in AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)] 
        fingerprints.append(fingerprint) 
        safe.append(mol_idx)
    except:
        print("Error",mol_idx) 
        continue
fingerprints=np.array(fingerprints)
x=fingerprints

# Prepare the target variable 'y' from the data frame
y=df.iloc[:,10:11]

# Function to build the neural network model
def build_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
    model = Sequential()
    model.add(InputLayer(input_shape=[2048])) # Input layer with shape [2048]

    # Add dense layers to the model based on the 'num_dense_layers' argument
    for i in range(num_dense_layers):
        model.add(Dense(units=num_dense_nodes, activation=activation))

    # Add the final output layer with 1 unit and a linear activation function
    model.add(Dense(units=1, activation='linear'))

    # Compile the model with Stochastic Gradient Descent optimizer and Mean Squared Error loss
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Define the search space for hyperparameters using Scikit-Optimize dimensions
dim_learning_rate = Real(low=1e-3, high=1e-1, prior='log-uniform', name='learning_rate')
dim_n_hidden = Integer(low=1, high=3, name='num_dense_layers')
dim_n_neurons = Integer(low=1000, high=5000, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'], name='activation')
dimensions = [dim_learning_rate, dim_n_hidden, dim_n_neurons, dim_activation]

# Default hyperparameters to be used if not provided
default_parameters = [1e-3, 1, 2000, 'relu']

# Function to generate the log directory name for TensorBoard logs
def log_dir_name(learning_rate, num_dense_layers, num_dense_nodes, activation):
    s = './my_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/'
    log_dir = s.format(learning_rate, num_dense_layers, num_dense_nodes, activation)
    return log_dir

path_best_model = 'best_model.keras'
@use_named_args(dimensions=dimensions)
def fitness(learning_rate,num_dense_layers, num_dense_nodes, activation):
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:',activation) 
    print()
    
    # Create the neural network with these hyper-parameters.
    model = build_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation)

    
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    # Use Keras to train the model.
    history = model.fit(x=x,
                        y=y,
                        epochs=3,
                        batch_size=128,
                        validation_data=(x,y),
                        callbacks=[callback_log])
    
    
     # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    mae = history.history['val_mae'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(mae))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    
    global best_mae
    best_mae = 0
    # If the classification accuracy of the saved model is improved ...
    if mae > best_mae:
        # Save the new model to harddisk.
        model.save(path_best_model)
        
        # Update the classification accuracy.
        best_mae = mae

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    tf.keras.backend.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -mae

# Measure the execution time of the optimization process
get_ipython().run_line_magic('time', '')

# Perform the optimization using Bayesian optimization with Gaussian Processes
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

# Plot the convergence of the optimization process
plot_convergence(search_result)

# Sort the results by the objective function value (MAE) and the corresponding hyperparameters
sorted_results = sorted(zip(search_result.func_vals, search_result.x_iters))

# Define the names of the hyperparameters for plotting
dim_names = ['learning_rate', 'num_dense_layers', 'num_dense_nodes', 'activation']

# Plot the objective function values for different hyperparameter combinations
fig, ax = plot_objective(result=search_result, dimensions=dim_names)
plt.savefig('all_dimen.png', dpi=400)

# Plot the evaluations of the objective function for different hyperparameter combinations
fig, ax = plot_evaluations(result=search_result, dimensions=dim_names)
