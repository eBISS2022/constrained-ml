#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import os
from ortools.linear_solver import pywraplp
import time
from scipy.optimize import curve_fit
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize



def click_through_rate(avg_ratings, num_reviews, dollar_ratings):
    dollar_rating_baseline = {"D": 3, "DD": 2, "DDD": 4, "DDDD": 4.5}
    return 1 / (1 + np.exp(
        np.array([dollar_rating_baseline[d] for d in dollar_ratings]) -
        avg_ratings * np.log1p(num_reviews) / 4))

def load_restaurant_data():
    def sample_restaurants(n):
        avg_ratings = np.random.uniform(1.0, 5.0, n)
        num_reviews = np.round(np.exp(np.random.uniform(0.0, np.log(200), n)))
        dollar_ratings = np.random.choice(["D", "DD", "DDD", "DDDD"], n)
        ctr_labels = click_through_rate(avg_ratings, num_reviews, dollar_ratings)
        return avg_ratings, num_reviews, dollar_ratings, ctr_labels


    def sample_dataset(n, testing_set):
        (avg_ratings, num_reviews, dollar_ratings, ctr_labels) = sample_restaurants(n)
        if testing_set:
            # Testing has a more uniform distribution over all restaurants.
            num_views = np.random.poisson(lam=3, size=n)
        else:
            # Training/validation datasets have more views on popular restaurants.
            num_views = np.random.poisson(lam=ctr_labels * num_reviews / 40.0, size=n)

        return pd.DataFrame({
                "avg_rating": np.repeat(avg_ratings, num_views),
                "num_reviews": np.repeat(num_reviews, num_views),
                "dollar_rating": np.repeat(dollar_ratings, num_views),
                "clicked": np.random.binomial(n=1, p=np.repeat(ctr_labels, num_views))
            })

    # Generate
    np.random.seed(42)
    data_train = sample_dataset(2000, testing_set=False)
    data_val = sample_dataset(1000, testing_set=False)
    data_test = sample_dataset(1000, testing_set=True)
    return data_train, data_val, data_test


def plot_ctr_truth(figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res)
    nrev = np.tile(np.linspace(0, 200, res), res)
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        drt = [drating] * (res*res)
        ctr = click_through_rate(avgr, nrev, drt)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('avrage rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_distribution(data, figsize=None):
    plt.figure(figsize=figsize)
    nbins = 15
    plt.subplot(131)
    plt.hist(data['avg_rating'], density=True, bins=nbins)
    plt.xlabel('average rating')
    plt.subplot(132)
    plt.hist(data['num_reviews'], density=True, bins=nbins)
    plt.xlabel('num. reviews')
    plt.subplot(133)
    vcnt = data['dollar_rating'].value_counts()
    vcnt /= vcnt.sum()
    plt.bar([0.5, 1.5, 2.5, 3.5],
            [vcnt['D'], vcnt['DD'], vcnt['DDD'], vcnt['DDDD']])
    plt.xlabel('dollar rating')
    plt.tight_layout()


def plot_ctr_estimation(estimator, scale,
        split_input=False, one_hot_categorical=True,
        figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    nrev = np.tile(np.linspace(0, 200, res), res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        if one_hot_categorical:
            # Categorical encoding for the dollar rating
            dr_cat = np.zeros((1, 4))
            dr_cat[0, i] = 1
            dr_cat = np.repeat((dr_cat), res*res, axis=0)
            # Concatenate all inputs
            x = np.hstack((avgr, nrev, dr_cat))
        else:
            # Integer encoding for the categorical attribute
            dr_cat = np.full((res*res, 1), i)
            x = np.hstack((avgr, nrev, dr_cat))
        # Split input, if requested
        if split_input:
            x = [x[:, i] for i in range(x.shape[1])]
        # Obtain the predictions
        ctr = estimator.predict(x)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('avrage rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_calibration(calibrators, scale, figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3

    # Average rating calibration
    avgr = np.linspace(0, 5, res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    avgr_cal = calibrators[0].predict(avgr)
    plt.subplot(131)
    plt.plot(avgr, avgr_cal)
    plt.xlabel('avg_rating')
    plt.ylabel('cal. output')
    # Num. review calibration
    nrev = np.linspace(0, 200, res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    nrev_cal = calibrators[1].predict(nrev)
    plt.subplot(132)
    plt.plot(nrev, nrev_cal)
    plt.xlabel('num_reviews')
    # Dollar rating calibration
    drating = np.arange(0, 4).reshape(-1, 1)
    drating_cal = calibrators[2].predict(drating).ravel()
    plt.subplot(133)
    xticks = np.linspace(0.5, 3.5, 4)
    plt.bar(xticks, drating_cal)
    plt.xticks(xticks, ['D', 'DD', 'DDD', 'DDDD'])

    plt.tight_layout()

# def euler_method(f, y0, t, return_gradients=False):
#     # Prepare a data structure for the results
#     y = np.zeros((len(t), len(y0)))
#     # Initial state
#     y[0, :] = y0
#     if return_gradients:
#         dy = np.zeros((len(t), len(y0)))
#     # Solve the ODE using Euler method
#     for i in range(1, len(t)):
#         # Current step and gradient
#         step = t[i] - t[i-1]
#         dy_l = f(y[i-1, :], t[i-1])
#         # If requested, store the gradient
#         if return_gradients:
#             dy[i-1, :] = dy_l
#         # Compute the next state
#         y[i, :] = y[i-1, :] + step * dy_l
#     # Return the results
#     if return_gradients:
#         return y, dy
#     else:
#         return y


# def plot_euler_method(y, t, dy=None, xlabel=None, ylabel=None,
#         figsize=None, horizon=2):
#     plt.figure(figsize=figsize)
#     plt.plot(t, y, marker='o', linestyle='')
#     # Plot gradients, if available
#     if dy is not None:
#         for i in range(len(y)-horizon):
#             ti, tf = t[i], t[i+horizon]
#             plt.plot([ti, tf], [y[i], y[i] + (tf-ti) * dy[i]],
#                     linestyle=':', color='0.2', alpha=0.5)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.tight_layout()
#     plt.show()


# def SIR(y, beta, gamma):
#     # Unpack the state
#     S, I, R = y
#     N = sum([S, I, R])
#     # Compute partial derivatives
#     dS = -beta * S * I / N
#     dI = beta * S * I / N - gamma * I
#     dR = gamma * I
#     # Return gradient
#     return np.array([dS, dI, dR])


# def plot_df_cols(data, figsize=None, legend=True):
#     # Build figure
#     fig = plt.figure(figsize=figsize)
#     # Setup x axis
#     x = data.index
#     plt.xlabel(data.index.name)
#     # Plot all columns
#     for cname in data.columns:
#         y = data[cname]
#         plt.plot(x, y, label=cname)
#     # Add legend
#     if legend and len(data.columns) <= 10:
#         plt.legend(loc='best')
#     # Make it compact
#     plt.tight_layout()
#     # Show
#     plt.show()


# def simulate_SIR(S0, I0, R0, beta, gamma, tmax, steps_per_day=1):
#     # Build initial state
#     Z = S0 + I0 + R0
#     Z = Z if Z > 0 else 1 # Handle division by zero
#     y0 = np.array([S0, I0, R0]) / Z
#     # Wrapper
#     nabla = lambda y, t: SIR(y, beta, gamma)
#     # Solve
#     t = np.linspace(0, tmax, steps_per_day * tmax + 1)
#     Y = odeint(nabla, y0, t)
#     # Wrap as dataframe
#     data = pd.DataFrame(data=Y, index=t, columns=['S', 'I', 'R'])
#     data.index.rename('time', inplace=True)
#     # Return the results
#     return data


# def simulate_RC(V0, tau, Vs, tmax, steps_per_unit=1):
#     # Define the initial state, gradient function, and time vector
#     y0 = np.array([V0])
#     nabla = lambda y, t: 1. / tau * (Vs - y)
#     t = np.linspace(0, tmax, steps_per_unit * tmax + 1)
#     # Solve
#     Y = odeint(nabla, y0, t)
#     # Wrap as dataframe
#     data = pd.DataFrame(data=Y, index=t, columns=['V'])
#     data.index.rename('time', inplace=True)
#     # Return the results
#     return data


# class SIRNablaLayer(keras.layers.Layer):
#     def __init__(self, beta_ref=0.1, gamma_ref=0.1,
#             fixed_beta=None, fixed_gamma=None):
#         super(SIRNablaLayer, self).__init__()
#         # Store the reference values/scales
#         self.beta_ref = beta_ref
#         self.gamma_ref = gamma_ref
#         # Prepare an initializer
#         p_init = tf.random_normal_initializer()
#         # Init the beta parameter
#         if fixed_beta is None:
#             self.logbeta = tf.Variable(
#                 initial_value=p_init(shape=(1, ), dtype="float32"),
#                 trainable=True,
#             )
#         else:
#             val = np.log(fixed_beta / self.beta_ref, dtype='float32')
#             self.logbeta = tf.Variable(initial_value=val, trainable=False)
#         # Init the gamma parameter
#         if fixed_gamma is None:
#             self.loggamma = tf.Variable(
#                 initial_value=p_init(shape=(1, ), dtype="float32"),
#                 trainable=True,
#             )
#         else:
#             val = np.log(fixed_gamma / self.gamma_ref, dtype='float32')
#             self.loggamma = tf.Variable(initial_value=val, trainable=False)

#     def get_beta(self):
#         return tf.math.exp(self.logbeta) * self.beta_ref

#     def get_gamma(self):
#         return tf.math.exp(self.loggamma) * self.gamma_ref

#     def call(self, inputs):
#         # Unpack the inputs (state and time)
#         y, t = inputs
#         # Slice the state
#         S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]
#         # Compute the gradient
#         N = tf.math.reduce_sum(y, axis=1, keepdims=True)
#         s2i = self.get_beta() * S * I / N
#         i2r = self.get_gamma() * I
#         dS = - s2i
#         dI = s2i - i2r
#         dR = i2r
#         # Concatenate
#         dy = tf.concat([dS, dI, dR], axis=1)
#         return dy


# class RCNablaLayer(keras.layers.Layer):
#     def __init__(self, tau_ref=0.1, vs_ref=0.1,
#             fixed_tau=None, fixed_vs=None):
#         super(RCNablaLayer, self).__init__()
#         # Store the reference values/scales
#         self.tau_ref = tau_ref
#         self.vs_ref = vs_ref
#         # Prepare an initializer
#         p_init = tf.random_normal_initializer()
#         # Init the tau parameter
#         if fixed_tau is None:
#             self.logtau = tf.Variable(
#                 initial_value=p_init(shape=(1, ), dtype="float32"),
#                 trainable=True,
#             )
#         else:
#             val = np.log(fixed_tau / self.tau_ref, dtype='float32')
#             self.logtau = tf.Variable(initial_value=val, trainable=False)
#         # Init the vs parameter
#         if fixed_vs is None:
#             self.logvs = tf.Variable(
#                 initial_value=p_init(shape=(1, ), dtype="float32"),
#                 trainable=True,
#             )
#         else:
#             val = np.log(fixed_vs / self.vs_ref, dtype='float32')
#             self.logvs = tf.Variable(initial_value=val, trainable=False)

#     def get_tau(self):
#         return tf.math.exp(self.logtau) * self.tau_ref

#     def get_vs(self):
#         return tf.math.exp(self.logvs) * self.vs_ref

#     def call(self, inputs):
#         # Unpack the inputs (state and time)
#         y, t = inputs
#         # Compute the gradient
#         dy = 1. / self.get_tau() * (self.get_vs() - y)
#         return dy


# class NPISIRNablaLayer(keras.layers.Layer):
#     def __init__(self, beta_pred, gamma_ref=0.1, fixed_gamma=None):
#         super(NPISIRNablaLayer, self).__init__()
#         # Store the model for predicting beta
#         self.beta_pred = beta_pred
#         # Store the reference values for gamma
#         self.gamma_ref = gamma_ref
#         # Prepare an initializer
#         p_init = tf.random_normal_initializer()
#         # Init the gamma parameter
#         if fixed_gamma is None:
#             self.loggamma = tf.Variable(
#                 initial_value=p_init(shape=(1, ), dtype="float32"),
#                 trainable=True,
#             )
#         else:
#             val = np.log(fixed_gamma / self.gamma_ref, dtype='float32')
#             self.loggamma = tf.Variable(initial_value=val, trainable=False)

#     def get_gamma(self):
#         return tf.math.exp(self.loggamma) * self.gamma_ref

#     def call(self, inputs):
#         # Unpack the inputs (state, time, and NPIs)
#         y, t, npis = inputs
#         # Slice the state
#         S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]
#         # Compute beta
#         beta = self.beta_pred(npis)
#         # Compute the gradient
#         N = tf.math.reduce_sum(y, axis=1, keepdims=True)
#         s2i = beta * S * I / N
#         i2r = self.get_gamma() * I
#         dS = - s2i
#         dI = s2i - i2r
#         dR = i2r
#         # Concatenate
#         dy = tf.concat([dS, dI, dR], axis=1)
#         return dy


# class ODEEulerModel(keras.Model):
#     def __init__(self, f, auxiliary_input=False, **params):
#         super(ODEEulerModel, self).__init__(**params)
#         # Store configuration parameters
#         self.f = f
#         self.auxiliary_input = auxiliary_input

#     def call(self, inputs, training=False):
#         # Unpack the initial state & time
#         if self.auxiliary_input:
#             y, T, aux = inputs
#         else:
#             y, T = inputs
#         # Solve the ODE via Euler's method
#         res = [y]
#         for i in range(T.shape[1]-1):
#             # Obtain vector with consecutive time steps
#             t, nt = T[:, i:i+1], T[:, i+1:i+2]
#             # Compute the state gradient
#             if self.auxiliary_input:
#                 dy = self.f([y, t, aux[:, i, :]], training=training)
#             else:
#                 dy = self.f([y, t], training=training)
#             # Update the state
#             y = y + (nt - t) * dy
#             # Store the result
#             res.append(y)
#         # Concatenate all results along a new axis
#         res = tf.stack(res, axis=1)
#         return res

#     def train_step(self, data):
#         # Unpack the data
#         if self.auxiliary_input:
#             (y0, T, aux), yt = data
#         else:
#             (y0, T), yt = data
#         # Loss computation
#         with tf.GradientTape() as tape:
#             # Integrate the ODE
#             if self.auxiliary_input:
#                 y = self.call([y0, T, aux], training=True)
#             else:
#                 y = self.call([y0, T], training=True)
#             # Compute the loss
#             mask = ~tf.math.is_nan(yt)
#             # residuals = y[mask] - yt[mask]
#             # loss = tf.math.reduce_mean(tf.math.square(residuals))
#             loss = self.compiled_loss(yt[mask], y[mask])
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # # Update main metrics
#         # self.metric_loss.update_state(loss)
#         # Update compiled metrics
#         self.compiled_metrics.update_state(yt[mask], y[mask])
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
#         # Unpack the data
#         if self.auxiliary_input:
#             (y0, T, aux), yt = data
#         else:
#             (y0, T), yt = data
#         # Integrate the ODE and compute the mask
#         if self.auxiliary_input:
#             y = self.call([y0, T, aux], training=False)
#         else:
#             y = self.call([y0, T], training=False)
#         # Updates the metrics tracking the loss
#         mask = ~tf.math.is_nan(yt)
#         self.compiled_loss(yt[mask], y[mask])
#         # Update the metrics
#         self.compiled_metrics.update_state(yt[mask], y[mask])
#         # Return a dict mapping metric names to current value.
#         # Note that it will include the loss (tracked in self.metrics).
#         return {m.name: m.result() for m in self.metrics}

    # @property
    # def metrics(self):
    #     # return [self.metric_loss]
    #     return [self.compiled_loss] + self.compiled_metrics.metrics


# def simulate_SIR_NPI(S0, I0, R0, beta, gamma, steps_per_day=1):
#     # Prepare the result data structure
#     S, I, R = [], [], []
#     # Loop over all weeks
#     res = []
#     S, I, R = S0, I0, R0
#     for w, b in enumerate(beta):
#         # Simulate one week
#         wres = simulate_SIR(S, I, R, b, gamma, 7,
#                 steps_per_day=steps_per_day)
#         # Retrieve all states
#         t = np.arange(0, 7)
#         wres_days = wres.loc[t]
#         wres_days['week'] = w
#         # Store the results
#         res.append(wres_days)
#         # Update the current state
#         S, I, R = wres[['S', 'I', 'R']].iloc[-1]
#     # Wrap into a dataframe
#     res = pd.concat(res, axis=0)
#     return res



# def gen_SIR_NPI_dataset(S0, I0, R0,
#         beta_base, gamma, npis, nweeks, steps_per_day=1, seed=None):
#     # Sample NPIs
#     npi_sched = sample_NPIs(npis, nweeks, seed=seed)
#     # Compute the corresponding beta values
#     beta = compute_beta(beta_base, npis, npi_sched)
#     # Simulate with the given beta schedule
#     beta_sched = [b for b in beta['beta'].values]
#     sir_data = simulate_SIR_NPI(S0, I0, R0, beta_sched, gamma, steps_per_day)
#     # Merge NPI data
#     res = sir_data.join(npi_sched, on='week')
#     res = res.join(beta, on='week')
#     # Reindex
#     idx = np.linspace(0, nweeks*7-1, nweeks*7)
#     res.set_index(idx, inplace=True)
#     return res


# def sample_points(ranges, n_samples, mode, seed=None):
#     assert(mode in ('uniform', 'lhs', 'max_min'))
#     # Build a space
#     space = Space(ranges)
#     # Seed the RNG
#     np.random.seed(seed)
#     # Sample
#     if mode == 'uniform':
#         X = space.rvs(n_samples)
#     elif mode == 'lhs':
#         lhs = Lhs(lhs_type="classic", criterion=None)
#         X = lhs.generate(space.dimensions, n_samples)
#     elif mode == 'max_min':
#         lhs = Lhs(criterion="maximin", iterations=100)
#         X = lhs.generate(space.dimensions, n_samples)
#     # Convert to an array
#     return np.array(X)


# def gen_SIR_input(max_samples, mode='lhs', normalize=True, seed=None):
#     # Sampling space: unnormalized S, I, R, plus beta
#     ranges = [(0.,1.), (0.,1.), (0.,1.)]
#     # Generate input
#     X = sample_points(ranges, max_samples, mode, seed=seed)
#     # Normalize
#     if normalize:
#         # Compute the normalization constants
#         Z = np.sum(X[:, :3], axis=1)
#         Z = np.where(Z > 0, Z, 1)
#         # Normalize the first three columns
#         X[:, :3] = X[:, :3] / Z.reshape(-1, 1)
#     # Wrap into a DataFrame
#     data = pd.DataFrame(data=X, columns=['S', 'I', 'R'])
#     return data


# def generate_SIR_output(sir_in, gamma, tmax, steps_per_day=1):
#     # Prepare a data structure for the results
#     res = []
#     # Loop over all examples
#     for idx, in_series in sir_in.iterrows():
#         # Unpack input
#         S0, I0, R0, beta = in_series
#         # Simulate
#         sim_data = simulate_SIR(S0, I0, R0, beta, gamma, tmax,
#                 steps_per_day=steps_per_day)
#         # Compute output
#         res.append(sim_data.values[-1, :])
#     # Wrap into a dataframe
#     data = pd.DataFrame(data=res, columns=['S', 'I', 'R'])
#     # Return
#     return data


# def plot_2D_samplespace(data, figsize=None):
#     # Build figure
#     fig = plt.figure(figsize=figsize)
#     # Setup axes
#     plt.xlabel('x_0')
#     plt.xlabel('x_1')
#     # Plot points
#     plt.plot(data[:, 0], data[:, 1], 'bo', label='samples', color='tab:blue')
#     plt.plot(data[:, 0], data[:, 1], 'bo', markersize=60, alpha=0.3,
#             color='tab:blue')
#     # Make it compact
#     plt.tight_layout()
#     # Show
#     plt.show()


def build_ml_model(input_size, output_size, hidden=[],
        output_activation='linear', scale=None, name=None):
    # Build all layers
    nn_in = keras.Input(input_size)
    nn_out = nn_in
    for h in hidden:
        nn_out = layers.Dense(h, activation='relu')(nn_out)
    nn_out = layers.Dense(output_size, activation=output_activation)(nn_out)
    if scale is not None:
        nn_out *= scale
    # Build the model
    model = keras.Model(inputs=nn_in, outputs=nn_out, name=name)
    return model


# class SimpleProgressBar(object):
#     def __init__(self, epochs, width=80):
#         self.epochs = epochs
#         self.width = width
#         self.csteps = 0

#     def __call__(self, epoch, logs):
#         # Compute the number of new steps
#         nsteps = int(self.width * epoch / self.epochs) - self.csteps
#         if nsteps > 0:
#             print('=' * nsteps, end='')
#         self.csteps += nsteps


def train_ml_model(model, X, y, epochs=20,
        verbose=0, patience=10, batch_size=32,
        validation_split=0.2, validation_data=None,
        sample_weight=None,
        eager_execution=False, loss='mse'):
    # Compile the model
    model.compile(optimizer='Adam', loss=loss,
            run_eagerly=eager_execution)
    # Build the early stop callback
    cb = []
    if validation_split > 0 or validation_data is not None:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # if verbose == 0:
    #     cb += [callbacks.LambdaCallback(on_epoch_end=
    #         SimpleProgressBar(epochs))]
    # Train the model
    history = model.fit(X, y,
                     validation_split=validation_split,
                     validation_data=validation_data,
                     callbacks=cb, batch_size=batch_size,
                     epochs=epochs, verbose=verbose,
                     sample_weight=sample_weight)
    return history



def plot_training_history(history=None,
        figsize=None, print_scores=True, restore_best_weights=True):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.tight_layout()
    plt.show()
    if print_scores:
        s = 'Model loss:'
        if 'val_loss' in history.history and restore_best_weights:
            bidx = np.argmin(history.history["val_loss"])
            vll = history.history["val_loss"][bidx]
            trl = history.history["loss"][bidx]
            s += f' {trl:.4f} (training)'
            s += f' {vll:.4f} (validation)'
        elif 'val_loss' in history.history and not restore_best_weights:
            vll = history.history["val_loss"][-1]
            trl = history.history["loss"][-1]
            s += f' {trl:.4f} (training)'
            s += f' {vll:.4f} (validation)'
        else:
            trl = history.history["loss"][-1]
            s += f' {trl:.4f} (training)'
        print(s)


# def get_ml_metrics(model, X, y):
#     # Obtain the predictions
#     pred = model.predict(X)
#     # Compute the root MSE
#     rmse = np.sqrt(metrics.mean_squared_error(y, pred))
#     # Compute the MAE
#     mae = metrics.mean_absolute_error(y, pred)
#     # Compute the coefficient of determination
#     r2 = metrics.r2_score(y, pred)
#     return r2, mae, rmse


# def save_ml_model(model, name):
#     model.save_weights(f'../data/{name}.h5')
#     with open(f'../data/{name}.json', 'w') as f:
#         f.write(model.to_json())


# def load_ml_model(name):
#     with open(f'../data/{name}.json') as f:
#         model = models.model_from_json(f.read())
#         model.load_weights(f'../data/{name}.h5')
#         return model


# class NPI(object):
#     def __init__(self, name, effect, cost):
#         self.name = name
#         self.effect = effect
#         self.cost = cost


# def compute_beta(beta_base, npis, npi_schedule):
#     # Build a starting vector
#     ns = len(npi_schedule)
#     beta = np.full((ns, ), beta_base)
#     # Loop over all the NPIs
#     for npi in npis:
#         # Build a vector with the effects
#         effect = np.where(npi_schedule[npi.name], npi.effect, 1)
#         # Apply the effect
#         beta *= effect
#     # Pack everything in a dataframe
#     res = pd.DataFrame(beta, index=npi_schedule.index, columns=['beta'])
#     return res



def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


# def sample_NPIs(npis, nweeks, seed=None):
#     # Build the set of all possible configurations
#     options = [np.array([0, 1]) for _ in range(len(npis))]
#     all_conf = cartesian_product(options)
#     idx = np.arange(len(all_conf))
#     # Seed the RNG
#     np.random.seed(seed)
#     # Sample some configurations at random
#     conf_idx = np.random.choice(idx, size=nweeks, replace=True)
#     conf = all_conf[conf_idx]
#     # Pack everything in a dataframe
#     res = pd.DataFrame(conf, columns=[npi.name for npi in npis])
#     for j, npi in enumerate(npis):
#         res[npi.name] = conf[:, j]
#     return res


# def solve_sir_brute_force(
#         npis : list,
#         S0 : float,
#         I0 : float,
#         R0 : float,
#         beta_base : float,
#         gamma : float,
#         nweeks : int = 1,
#         budget : float = None):
#     # Build a table with beta-confs for a week
#     doms = [np.array([0, 1]) for _ in range(len(npis))]
#     beta_confs = cartesian_product(doms)
#     # Compute costs
#     npi_costs = np.array([npi.cost for npi in npis])
#     beta_costs = np.dot(beta_confs, npi_costs)
#     # Compute beta values
#     npi_eff = np.array([npi.effect for npi in npis])
#     beta_vals = beta_confs * npi_eff
#     beta_vals += (1-beta_confs)
#     beta_vals = beta_base * np.product(beta_vals, axis=1)
#     # Build all combinations of beta-values and costs
#     bsched_cost = cartesian_product([beta_costs for _ in range(nweeks)])
#     bsched_vals = cartesian_product([beta_vals for _ in range(nweeks)])
#     # Filter out configurations that do not meen the budget
#     mask = np.sum(bsched_cost, axis=1) <= budget
#     bsched_feas = bsched_vals[mask]
#     # Simulate them all
#     best_S = -np.inf
#     best_sched = None
#     for bsched in bsched_feas:
#         # Simulate
#         res = simulate_SIR_NPI(S0, I0, R0, bsched, gamma, steps_per_day=1)
#         last_S = res.iloc[-1]['S']
#         if last_S > best_S:
#             best_S = last_S
#             best_sched = bsched
#     return best_S, best_sched



# def solve_sir_planning(
#         keras_model,
#         npis : list,
#         S0 : float,
#         I0 : float,
#         R0 : float,
#         beta_base : float,
#         nweeks : int = 1,
#         budget : float = None,
#         tlim : float = None,
#         init_state_csts : bool = True,
#         network_csts : bool = True,
#         effectiveness_csts : bool = True,
#         cost_csts : bool = True,
#         beta_ub_csts : bool = True,
#         use_hints : bool = True):

#     assert(0 <= beta_base <= 1)

#     # Build a model object
#     slv = pywraplp.Solver.CreateSolver('CBC')

#     # Define the variables
#     X = {}
#     for t in range(nweeks+1):
#         # Build the SIR variables
#         X['S', t] = slv.NumVar(0, 1, f'S_{t}')
#         X['I', t] = slv.NumVar(0, 1, f'I_{t}')
#         X['R', t] = slv.NumVar(0, 1, f'R_{t}')
#         if t < nweeks:
#             X['b', t] = slv.NumVar(0, 1, f'b_{t}')

#         # Build the NPI variables
#         if t < nweeks:
#             for npi in npis:
#                 name = npi.name
#                 X[name, t] = slv.IntVar(0, 1, f'{name}_{t}')

#     # Build the cost variable
#     maxcost = sum(npi.cost for npi in npis) * nweeks
#     X['cost'] = slv.NumVar(0, maxcost, 'cost')

#     # Build the initial state constraints
#     if init_state_csts:
#         slv.Add(X['S', 0] == S0)
#         slv.Add(X['I', 0] == I0)
#         slv.Add(X['R', 0] == R0)

#     # Build the network constraints
#     if network_csts:
#         # Build a backend object
#         bkd = ortools_backend.OrtoolsBackend()
#         # Convert the keras model in internal format
#         nn = keras_reader.read_keras_sequential(keras_model)
#         # Set bounds
#         nn.layer(0).update_lb(np.zeros(4))
#         nn.layer(0).update_ub(np.ones(4))
#         # Propagate bounds
#         ibr_bounds(nn)
#         # Build the encodings
#         for t in range(1, nweeks+1):
#             vin = [X['S',t-1], X['I',t-1], X['R',t-1], X['b',t-1]]
#             vout = [X['S',t], X['I',t], X['R',t]]
#             encode(bkd, nn, slv, vin, vout, f'nn_{t}')

#     # Build the effectiveness constraints
#     if effectiveness_csts:
#         for t in range(nweeks):
#             # Set base beta as the starting one
#             bbeta = beta_base
#             # Process all NPIs
#             for i, npi in enumerate(npis):
#                 name = npi.name
#                 # For all NPIs but the last, build a temporary beta variable
#                 if i < len(npis)-1:
#                     # Build a new variable to be used as current beta
#                     cbeta = slv.NumVar(0, 1, f'b_{name}_{t}')
#                     X[f'b_{name}', t] = cbeta
#                 else:
#                     # Use the "real" beta as current
#                     cbeta = X['b', t]
#                 # Linearize a guarded division
#                 slv.Add(cbeta >= npi.effect * bbeta - 1 + X[name, t])
#                 slv.Add(cbeta >= bbeta - X[name, t])
#                 # Add an upper bound, if requested
#                 if beta_ub_csts:
#                     slv.Add(cbeta <= npi.effect * bbeta + 1 - X[name, t])
#                     slv.Add(cbeta <= bbeta + X[name, t])
#                 # Reset base beta
#                 bbeta = cbeta

#     # Define the cost
#     if cost_csts:
#         slv.Add(X['cost'] == sum(npi.cost * X[npi.name, t]
#             for t in range(nweeks) for npi in npis))
#         if budget is not None:
#             slv.Add(X['cost'] <= budget)

#     # Define the objectives
#     slv.Maximize(X['S', nweeks])

#     # Build a heuristic solution
#     if use_hints:
#         hvars, hvals = [], []
#         # Sort NPIs by decreasing "efficiency"
#         snpis = sorted(npis, key=lambda npi: -(1-npi.effect) / npi.cost)
#         # Loop over all the NPIs
#         rbudget = budget
#         for npi in snpis:
#             # Activate on as many weeks as possible
#             for w in range(nweeks):
#                 if rbudget > npi.cost:
#                     hvars.append(X[npi.name, w])
#                     hvals.append(1)
#                     rbudget -= npi.cost
#                 else:
#                     hvars.append(X[npi.name, w])
#                     hvals.append(0)
#         # Set hints
#         slv.SetHint(hvars, hvals)

#     # Set a time limit
#     if tlim is not None:
#         slv.SetTimeLimit(tlim * 1000)
#     # Solve the problem
#     status = slv.Solve()
#     # Return the result
#     res = None
#     closed = False
#     if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
#         res = {}
#         for k, x in X.items():
#             res[k] = x.solution_value()
#     if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.INFEASIBLE):
#         closed = True
#     return res, closed



# def sol_to_dataframe(sol, npis, nweeks):
#     # Define the column names
#     cols = ['S', 'I', 'R', 'b']
#     cols += [npi.name for npi in npis]
#     # Prepare a result dataframe
#     res = pd.DataFrame(index=range(nweeks+1), columns=cols, data=np.nan)
#     for w in range(nweeks):
#         res.loc[w, 'S'] = sol['S', w]
#         res.loc[w, 'I'] = sol['I', w]
#         res.loc[w, 'R'] = sol['R', w]
#         res.loc[w, 'b'] = sol['b', w]
#         for n in npis:
#             res.loc[w, n.name] = sol[n.name, w]
#     # Store the state for the final week
#     res.loc[nweeks, 'S'] = sol['S', nweeks]
#     res.loc[nweeks, 'I'] = sol['I', nweeks]
#     res.loc[nweeks, 'R'] = sol['R', nweeks]
#     # Return the result
#     return res


# def generate_market_dataset(nsamples, nitems, noise=0, seed=None):
#     assert(0 <= noise <= 1)
#     # Seed the RNG
#     np.random.seed(seed)
#     # Generate costs
#     speed = np.random.choice([10, 13, 15], size=nitems)
#     base = 0.4 * np.random.rand(nitems)
#     scale = 0.4 + 1 * np.random.rand(nitems)
#     offset = -0.7 * np.random.rand(nitems)

#     # Generate input
#     x = np.random.rand(nsamples)
#     # Prepare a result dataset
#     res = pd.DataFrame(data=x, columns=['x'])

#     # scale = np.sort(scale)[::-1]
#     for i in range(nitems):
#         # Compute base cost
#         cost = scale[i] / (1 + np.exp(-speed[i] * (x + offset[i])))
#         # Rebase
#         cost = cost - np.min(cost) + base[i]
#         # sx = direction[i]*speed[i]*(x+offset[i])
#         # cost = base[i] + scale[i] / (1 + np.exp(sx))
#         res[f'C{i}'] = cost
#     # Add noise
#     if noise > 0:
#         for i in range(nitems):
#             pnoise = noise * np.random.randn(nsamples)
#             res[f'C{i}'] = np.maximum(0, res[f'C{i}'] + pnoise)
#     # Reindex
#     res.set_index('x', inplace=True)
#     # Sort by index
#     res.sort_index(inplace=True)
#     # Return results
#     return res



def split_datasets(dslist, fraction, seed=None, standardize=None):
    assert(0 < fraction < 1)
    assert(all(len(ds) == len(dslist[0]) for ds in dslist))
    # Seed the RNG
    np.random.seed(seed)
    # Shuffle the indices
    idx = np.arange(len(dslist[0]))
    np.random.shuffle(idx)
    # Partition the indices
    sep = int(len(dslist[0]) * (1-fraction))
    # Partition the datasets
    res_a, res_b, scalers = [], [], []
    for ds in dslist:
        # Split the datasets
        a = ds.iloc[idx[:sep]].copy()
        b = ds.iloc[idx[sep:]].copy()
        # Apply standardization
        if standardize is not None:
            nstd = len(standardize)
            scl = StandardScaler()
            a[standardize] = scl.fit_transform(a[standardize].values.reshape(-1, nstd))
            b[standardize] = scl.transform(b[standardize].values.reshape(-1, nstd))
            scalers.append(scl)
        # Store the datasets
        res_a.append(a)
        res_b.append(b)
    # Apply standardization, if requested
    if standardize is None:
        return res_a, res_b
    else:
        return res_a, res_b, scalers


def load_cmapss_data(data_folder):
    # Read the CSV files
    fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    # data.columns = cols
    return data


# def generate_market_problem(nitems, rel_req, seed=None):
#     # Seed the RNG
#     np.random.seed(seed)
#     # Generate the item values
#     # values = np.ones(nitems)
#     values = 1 + 0.2*np.random.rand(nitems)
#     # Generate the requirement
#     req = rel_req * np.sum(values)
#     # Return the results
#     return MarketProblem(values, req)


# class MarketProblem(object):
#     """Docstring for MarketProblem. """

#     def __init__(self, values, requirement):
#         """TODO: to be defined. """
#         # Store the problem configuration
#         self.values = values
#         self.requirement = requirement

#     def solve(self, costs, tlim=None, print_solution=False):
#         # Quick access to some useful fields
#         values = self.values
#         req = self.requirement
#         nv = len(values)
#         # Build the solver
#         slv = pywraplp.Solver.CreateSolver('CBC')
#         # Build the variables
#         x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
#         # Build the requirement constraint
#         rcst = slv.Add(sum(values[i] * x[i] for i in range(nv)) >= req)
#         # Build the objective
#         slv.Minimize(sum(costs[i] * x[i] for i in range(nv)))

#         # Set a time limit, if requested
#         if tlim is not None:
#             slv.SetTimeLimit(1000 * tlim)
#         # Solve the problem
#         status = slv.Solve()
#         # Prepare the results
#         if status in (slv.OPTIMAL, slv.FEASIBLE):
#             res = []
#             # Extract the solution
#             sol = [x[i].solution_value() for i in range(nv)]
#             res.append(sol)
#             # Determine whether the problem was closed
#             if status == slv.OPTIMAL:
#                 res.append(True)
#             else:
#                 res.append(False)
#         else:
#             # TODO I am not handling the unbounded case
#             # It should never arise in the first place
#             if status == slv.INFEASIBLE:
#                 res = [None, True]
#             else:
#                 res = [None, False]
#         # Print the solution, if requested
#         if print_solution:
#             print_sol(self, res[0], res[1], costs)
#         return res

#     def __repr__(self):
#         return f'<MarketProblem: {self.values} {self.requirement}>'


# def print_sol(prb, sol, closed, costs):
#     # Obtain indexes of selected items
#     idx = [i for i in range(len(sol)) if sol[i] > 0]
#     # Obtain the corresponding values
#     values = [prb.values[i] for i in idx]
#     # Print selected items with values and costs
#     s = ', '.join(f'{i}' for i in idx)
#     print('Selected items:', s)
#     s = f'Cost: {sum(costs):.2f}, '
#     s += f'Value: {sum(values):.2f}, '
#     s += f'Requirement: {prb.requirement:.2f}, '
#     s += f'Closed: {closed}'
#     print(s)


# def compute_regret(problem, predictor, pred_in, true_costs, tlim=None):
#     # Obtain all predictions
#     costs = predictor.predict(pred_in)
#     # Compute all solutions
#     sols = []
#     for c in costs:
#         sol, _ = problem.solve(c, tlim=tlim)
#         sols.append(sol)
#     sols = np.array(sols)
#     # Compute the true solutions
#     tsols = []
#     for c in true_costs:
#         sol, _ = problem.solve(c, tlim=tlim)
#         tsols.append(sol)
#     tsols = np.array(tsols)
#     # Compute true costs
#     costs_with_predictions = np.sum(true_costs * sols, axis=1)
#     costs_with_true_solutions = np.sum(true_costs * tsols, axis=1)
#     # Compute regret
#     regret = costs_with_predictions - costs_with_true_solutions
#     # Return true costs
#     return regret


# def plot_histogram(data, label=None, bins=20, figsize=None,
#         data2=None, label2=None, print_mean=False):
#     # Build figure
#     fig = plt.figure(figsize=figsize)
#     # Setup x axis
#     plt.xlabel(label)
#     # Define bins
#     rmin, rmax = data.min(), data.max()
#     if data2 is not None:
#         rmin = min(rmin, data2.min())
#         rmax = max(rmax, data2.max())
#     bins = np.linspace(rmin, rmax, bins)
#     # Histogram
#     hist, edges = np.histogram(data, bins=bins)
#     hist = hist / np.sum(hist)
#     plt.step(edges[:-1], hist, where='post', label=label)
#     if data2 is not None:
#         hist2, edges2 = np.histogram(data2, bins=bins)
#         hist2 = hist2 / np.sum(hist2)
#         plt.step(edges2[:-1], hist2, where='post', label=label2)
#     # Make it compact
#     plt.tight_layout()
#     # Legend
#     plt.legend()
#     # Show
#     plt.show()
#     # Print mean, if requested
#     if print_mean:
#         s = f'Mean: {np.mean(data):.3f}'
#         if label is not None:
#             s += f' ({label})'
#         if data2 is not None:
#             s += f', {np.mean(data2):.3f}'
#             if label2 is not None:
#                 s += f' ({label2})'
#         print(s)


# def print_ml_metrics(model, X, y, label=None):
#     # Obtain the predictions
#     pred = model.predict(X)
#     # Compute the root MSE
#     rmse = np.sqrt(metrics.mean_squared_error(y, pred))
#     # Compute the MAE
#     mae = metrics.mean_absolute_error(y, pred)
#     # Compute the coefficient of determination
#     r2 = metrics.r2_score(y, pred)
#     lbl = '' if label is None else f' ({label})'
#     print(f'R2: {r2:.2f}, MAE: {mae:.2}, RMSE: {rmse:.2f}{lbl}')


# class DFLModel(keras.Model):
#     def __init__(self, prb, tlim=None, recompute_chance=1, **params):
#         super(DFLModel, self).__init__(**params)
#         # Store configuration parameters
#         self.prb = prb
#         self.tlim = tlim
#         self.recompute_chance = recompute_chance
#         # Build metrics
#         self.metric_loss = keras.metrics.Mean(name="loss")
#         # self.metric_regret = keras.metrics.Mean(name="regret")
#         # self.metric_mae = keras.metrics.MeanAbsoluteError(name="mae")
#         # Prepare a field for the solutions
#         self.sol_store = None

#     # def dfl_fit(self, X, y, **kwargs):
#     #     # Precompute all solutions for the true costs
#     #     self.sol_store = []
#     #     for c in y:
#     #         sol, closed = self.prb.solve(c, tlim=self.tlim)
#     #         self.sol_store.append(sol)
#     #     self.sol_store = np.array(self.sol_store)
#     #     # Call the normal fit method
#     #     return self.fit(X, y, **kwargs)


#     def fit(self, X, y, **kwargs):
#         # Precompute all solutions for the true costs
#         self.sol_store = []
#         for c in y:
#             sol, closed = self.prb.solve(c, tlim=self.tlim)
#             self.sol_store.append(sol)
#         self.sol_store = np.array(self.sol_store)
#         # Call the normal fit method
#         return super(DFLModel, self).fit(X, y, **kwargs)

#     def _find_best(self, costs):
#         tc = np.dot(self.sol_store, costs)
#         best_idx = np.argmin(tc)
#         best = self.sol_store[best_idx]
#         return best

#     def train_step(self, data):
#         # Unpack the data
#         x, costs_true = data
#         # Quick access to some useful fields
#         prb = self.prb
#         tlim = self.tlim

#         # Loss computation
#         with tf.GradientTape() as tape:
#             # Obtain the predictions
#             costs = self(x, training=True)
#             # Solve all optimization problems
#             sols, tsols = [], []
#             for c, tc in zip(costs.numpy(), costs_true.numpy()):
#                 # Decide whether to recompute the solution
#                 if np.random.rand() < self.recompute_chance:
#                     sol, closed = prb.solve(c, tlim=self.tlim)
#                     # Store the solution, if needed
#                     if self.recompute_chance < 1:
#                         # Check if the solutions is already stored
#                         if not (self.sol_store == sol).all(axis=1).any():
#                             self.sol_store = np.vstack((self.sol_store, sol))
#                 else:
#                     sol = self._find_best(c)
#                 # Find the best solution with the predicted costs
#                 sols.append(sol)
#                 # Find the best solution with the true costs
#                 tsol = self._find_best(tc)
#                 tsols.append(tsol)
#             # Convert solutions to numpy arrays
#             sols = np.array(sols)
#             tsols = np.array(tsols)
#             # Compute the cost difference
#             cdiff = costs - costs_true
#             # Compute the solution difference
#             sdiff = tsols - sols
#             # Compute the loss terms
#             loss_terms = tf.reduce_sum(cdiff * sdiff, axis=1)
#             # Compute the mean loss
#             loss = tf.reduce_mean(loss_terms)

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update main metrics
#         self.metric_loss.update_state(loss)
#         # regrets = tf.reduce_sum((sols - tsols) * costs_true, axis=1)
#         # mean_regret = tf.reduce_mean(regrets)
#         # self.metric_regret.update_state(mean_regret)
#         # self.metric_mae.update_state(costs_true, costs)
#         # Update compiled metrics
#         self.compiled_metrics.update_state(costs_true, costs)
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}

#     # def test_step(self, data):
#     #     # Unpack the data
#     #     x, costs_true = data
#     #     # Compute predictions
#     #     costs = self(x, training=False)
#     #     # Updates the metrics tracking the loss
#     #     self.compiled_loss(y, y_pred, regularization_losses=self.losses)
#     #     # Update the metrics.
#     #     self.compiled_metrics.update_state(y, y_pred)
#     #     # Return a dict mapping metric names to current value.
#     #     # Note that it will include the loss (tracked in self.metrics).
#     #     return {m.name: m.result() for m in self.metrics}

#     @property
#     def metrics(self):
#         return [self.metric_loss]
#         # return [self.metric_loss, self.metric_regret]


# def build_dfl_ml_model(input_size, output_size,
#         problem, tlim=None, hidden=[], recompute_chance=1,
#         output_activation='linear', name=None):
#     # Build all layers
#     nnin = keras.Input(input_size)
#     nnout = nnin
#     for h in hidden:
#         nnout = layers.Dense(h, activation='relu')(nnout)
#     nnout = layers.Dense(output_size, activation=output_activation)(nnout)
#     # Build the model
#     model = DFLModel(problem, tlim=tlim, recompute_chance=recompute_chance,
#             inputs=nnin, outputs=nnout, name=name)
#     return model


# def train_dfo_model(model, X, y, tlim=None,
#         epochs=20, verbose=0, patience=10, batch_size=32,
#         validation_split=0.2):
#     # Compile and train
#     model.compile(optimizer='Adam', run_eagerly=True)
#     if validation_split > 0:
#         cb = [callbacks.EarlyStopping(patience=patience,
#             restore_best_weights=True)]
#     else:
#         cb = None
#     # history = model.dfl_fit(X, y, validation_split=validation_split,
#     #                  callbacks=cb, batch_size=batch_size,
#     #                  epochs=epochs, verbose=verbose)
#     history = model.fit(X, y, validation_split=validation_split,
#                      callbacks=cb, batch_size=batch_size,
#                      epochs=epochs, verbose=verbose)
#     return history




# def load_cmapss_data(data_folder):
#     # Read the CSV files
#     fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
#     cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
#     datalist = []
#     nmcn = 0
#     for fstem in fnames:
#         # Read data
#         data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
#         # Drop the last two columns (parsing errors)
#         data.drop(columns=[26, 27], inplace=True)
#         # Replace column names
#         data.columns = cols
#         # Add the data source
#         data['src'] = fstem
#         # Shift the machine numbers
#         data['machine'] += nmcn
#         nmcn += len(data['machine'].unique())
#         # Generate RUL data
#         cnts = data.groupby('machine')[['cycle']].count()
#         cnts.columns = ['ftime']
#         data = data.join(cnts, on='machine')
#         data['rul'] = data['ftime'] - data['cycle']
#         data.drop(columns=['ftime'], inplace=True)
#         # Store in the list
#         datalist.append(data)
#     # Concatenate
#     data = pd.concat(datalist)
#     # Put the 'src' field at the beginning and keep 'rul' at the end
#     data = data[['src'] + cols + ['rul']]
#     # data.columns = cols
#     return data


# def split_by_field(data, field):
#     res = {}
#     for fval, gdata in data.groupby(field):
#         res[fval] = gdata
#     return res


# def plot_df_heatmap(data, labels=None, vmin=-1.96, vmax=1.96,
#         figsize=None, s=4, normalize='standardize'):
#     # Normalize the data
#     if normalize == 'standardize':
#         data = data.copy()
#         data = (data - data.mean()) / data.std()
#     else:
#         raise ValueError('Unknown normalization method')
#     # Build a figure
#     plt.figure(figsize=figsize)
#     plt.imshow(data.T.iloc[:, :], aspect='auto',
#             cmap='RdBu', vmin=vmin, vmax=vmax)
#     if labels is not None:
#         # nonzero = data.index[labels != 0]
#         ncol = len(data.columns)
#         lvl = - 0.05 * ncol
#         # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
#         #         s=s, color='tab:orange')
#         plt.scatter(labels.index, np.ones(len(labels)) * lvl,
#                 s=s,
#                 color=plt.get_cmap('tab10')(np.mod(labels, 10)))
#     plt.tight_layout()
#     plt.show()


# def partition_by_machine(data, tr_machines):
#     # Separate
#     tr_machines = set(tr_machines)
#     tr_list, ts_list = [], []
#     for mcn, gdata in data.groupby('machine'):
#         if mcn in tr_machines:
#             tr_list.append(gdata)
#         else:
#             ts_list.append(gdata)
#     # Collate again
#     tr_data = pd.concat(tr_list)
#     ts_data = pd.concat(ts_list)
#     return tr_data, ts_data


# def plot_pred_scatter(y_pred, y_true, figsize=None, autoclose=True):
#     plt.figure(figsize=figsize)
#     plt.scatter(y_pred, y_true, marker='.', alpha=max(0.01, 1 / len(y_pred)))
#     xl, xu = plt.xlim()
#     yl, yu = plt.ylim()
#     l, u = min(xl, yl), max(xu, yu)
#     plt.plot([l, u], [l, u], ':', c='0.3')
#     plt.xlim(l, u)
#     plt.ylim(l, u)
#     plt.xlabel('prediction')
#     plt.ylabel('target')
#     plt.tight_layout()
#     plt.show()


# def rescale_CMAPSS(tr, ts):
#     # Define input columns
#     dt_in = tr.columns[3:-1]
#     # Compute mean and standard deviation
#     trmean = tr[dt_in].mean()
#     trstd = tr[dt_in].std().replace(to_replace=0, value=1) # handle static fields
#     # Rescale all inputs
#     ts_s = ts.copy()
#     ts_s[dt_in] = (ts_s[dt_in] - trmean) / trstd
#     tr_s = tr.copy()
#     tr_s[dt_in] = (tr_s[dt_in] - trmean) / trstd
#     # Compute the maximum RUL
#     trmaxrul = tr['rul'].max()
#     # Normalize the RUL
#     ts_s['rul'] = ts['rul'] / trmaxrul
#     tr_s['rul'] = tr['rul'] / trmaxrul
#     # Return results
#     params = {'trmean': trmean, 'trstd': trstd, 'trmaxrul': trmaxrul}
#     return tr_s, ts_s, params


# def plot_rul(pred=None, target=None,
#         stddev=None,
#         q1_3=None,
#         same_scale=True,
#         figsize=None):
#     plt.figure(figsize=figsize)
#     if target is not None:
#         plt.plot(range(len(target)), target, label='target',
#                 color='tab:orange')
#     if pred is not None:
#         if same_scale or target is None:
#             ax = plt.gca()
#         else:
#             ax = plt.gca().twinx()
#         ax.plot(range(len(pred)), pred, label='pred',
#                 color='tab:blue')
#         if stddev is not None:
#             ax.fill_between(range(len(pred)),
#                     pred-stddev, pred+stddev,
#                     alpha=0.3, color='tab:blue', label='+/- std')
#         if q1_3 is not None:
#             ax.fill_between(range(len(pred)),
#                     q1_3[0], q1_3[1],
#                     alpha=0.3, color='tab:blue', label='1st/3rd quartile')
#     plt.xlabel('time')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# class RULCostModel:
#     def __init__(self, maintenance_cost, safe_interval=0):
#         self.maintenance_cost = maintenance_cost
#         self.safe_interval = safe_interval

#     def cost(self, machine, pred, thr, return_margin=False):
#         # Merge machine and prediction data
#         tmp = np.array([machine, pred]).T
#         tmp = pd.DataFrame(data=tmp,
#                            columns=['machine', 'pred'])
#         # Cost computation
#         cost = 0
#         nfails = 0
#         slack = 0
#         for mcn, gtmp in tmp.groupby('machine'):
#             idx = np.nonzero(gtmp['pred'].values < thr)[0]
#             if len(idx) == 0:
#                 cost += self.maintenance_cost
#                 nfails += 1
#             else:
#                 cost -= max(0, idx[0] - self.safe_interval)
#                 slack += len(gtmp) - idx[0]
#         if not return_margin:
#             return cost
#         else:
#             return cost, nfails, slack


# def optimize_threshold(machine, pred, th_range, cmodel,
#         plot=False, figsize=None):
#     # Compute the optimal threshold
#     costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
#     opt_th = th_range[np.argmin(costs)]
#     # Plot
#     if plot:
#         plt.figure(figsize=figsize)
#         plt.plot(th_range, costs)
#         plt.xlabel('threshold')
#         plt.ylabel('cost')
#         plt.tight_layout()
#     # Return the threshold
#     return opt_th


# def plot_series(series=None, samples=None, std=None, target=None,
#         figsize=None, s=4, alpha=0.95, xlabel=None, ylabel=None,
#         samples_lbl='samples', target_lbl='target',
#         samples2=None, samples2_lbl=None):
#     plt.figure(figsize=figsize)
#     if series is not None:
#         plt.plot(series.index, series, label=ylabel)
#     if target is not None:
#         plt.plot(target.index, target, ':', label=target_lbl, color='0.7')
#     if std is not None:
#         plt.fill_between(series.index,
#                 series.values - std, series.values + std,
#                 alpha=0.3, color='tab:blue', label='+/- std')
#     if samples is not None:
#         plt.scatter(samples.index, samples, label=samples_lbl,
#                 color='tab:orange')
#     if samples2 is not None:
#         plt.scatter(samples2.index, samples2, label=samples2_lbl,
#                 color='tab:red')
#     if xlabel is not None:
#         plt.xlabel(xlabel)
#     if (series is not None) + (samples is not None) + \
#             (target is not None) > 1:
#         plt.legend()
#     else:
#         plt.ylabel(ylabel)
#     plt.tight_layout()


# def max_acq(mu, std, best_y, n_samples, ftype, alpha=0):
#     # Obtain the acquisition function values
#     if ftype == 'pi':
#         acq = stats.norm.cdf(best_y, loc=mu, scale=std)
#         acq = np.array(acq)
#     elif ftype == 'lcb':
#         lcb, ucb = stats.norm.interval(alpha, loc=mu, scale=std)
#         acq = -lcb
#     elif ftype == 'ei':
#         acq = (best_y - mu) * stats.norm.cdf(best_y, loc=mu, scale=std)
#         acq += std * stats.norm.pdf(best_y, loc=mu, scale=std)
#         acq = np.array(acq)
#     else:
#         raise ValueError('Unknown acquisition function type')
#     # Return the best point
#     bidx = int(np.argmax(acq))
#     best_x = mu.index[bidx]
#     best_acq = acq[bidx]
#     return best_x, best_acq


# def univariate_gp_tt(kernel, x_tr, y_tr, x_eval,
#         suppress_warning=False):
#     assert(len(x_eval.shape) == 1 or x_eval.shape[1] == 1)
#     assert(len(x_tr.shape) == 1 or x_tr.shape[1] == 1)
#     # Reshape x_tr and x_eval
#     x_tr = x_tr.reshape(-1, 1)
#     x_eval = x_eval.ravel()
#     # Build a GP model
#     mdl = GaussianProcessRegressor(kernel=kernel,
#                                    n_restarts_optimizer=10,
#                                    normalize_y=True)
#     # Traing the GP model
#     with warnings.catch_warnings():
#         if suppress_warning:
#             warnings.simplefilter('ignore')
#         mdl.fit(x_tr, y_tr)
#     # Predict
#     mu, std = mdl.predict(x_eval.reshape(-1, 1), return_std=True)
#     # Return results
#     mu = pd.Series(index=x_eval, data=mu.ravel())
#     std = pd.Series(index=x_eval, data=std.ravel())
#     return mu, std


# def simple_univariate_BO(f, l, u, max_it=10, init_points=3, alpha=0.95,
#         seed=None, n_samples_gs=10000, ftype='lcb', return_state=False,
#         tol=1e-3, suppress_warnings=False):
#     # Define the kernel
#     kernel = RBF(1, (1e-6, 1e0)) + WhiteKernel(1, (1e-6, 1e0))
#     # Reseed the RNG
#     np.random.seed(seed)
#     # Sample initial points
#     X = np.random.uniform(l, u, size=(init_points, 1))
#     y = f(X)
#     t = [0] * init_points
#     # Determine the current best point
#     best_idx = int(np.argmin(y))
#     best_x = X[best_idx][0]
#     best_y = y[best_idx]
#     # Init the GP model
#     x = np.linspace(l, u, n_samples_gs)
#     mu, std = univariate_gp_tt(kernel, X, y, x, suppress_warnings)
#     # Main Loop
#     for nit in range(max_it):
#         # Minimize the acquisition function
#         next_x, acq = max_acq(mu, std, best_y, n_samples_gs, ftype, alpha)
#         # Check termination criteria
#         if acq < tol: break
#         # Update the best solution
#         next_y = f(next_x)
#         if next_y < best_y:
#             best_x = next_x
#             best_y = next_y
#         # Udpate the pool and the GP model
#         if nit < max_it - 1:
#             X = np.vstack((X, next_x))
#             y = np.vstack((y, f(next_x)))
#             t.append(nit+1)
#             mu, std = univariate_gp_tt(kernel, X, y, x, suppress_warnings)
#     # Return results
#     res = best_x
#     if return_state:
#         samples = pd.Series(index=X.ravel(), data=y.ravel())
#         sopt = pd.Series(index=[best_x], data=[best_y])
#         res = res, {'samples': samples, 'mu': mu, 'std': std, 't': t,
#                 'sopt': sopt}
#     return res


# def opt_classifier_policy(mdl, X, y, machines, cmodel,
#         n_iter=10, init_points=5, epochs_per_it=3,
#         validation_split=0.2, seed=None, verbose=0):
#     # Store weights
#     init_wgt = mdl.get_weights()
#     # Define a data structre to store the weights for each solution
#     stored_weights = {}

#     # Define the cost function
#     def f(x):
#         # Unpack the input
#         thr, c0_weight = x
#         # Define new labels
#         tr_lbl = (y >= thr)
#         # Reset weights
#         mdl.set_weights(init_wgt)
#         # Define sample weigths
#         sample_weight = np.where(tr_lbl, 1, c0_weight)
#         # Fit
#         start = time.time()
#         train_ml_model(mdl, X, tr_lbl, epochs=epochs_per_it,
#                 validation_split=validation_split,
#                 sample_weight=sample_weight, loss='binary_crossentropy')
#         dur = time.time() - start
#         # Cost computation
#         tr_pred = np.round(mdl.predict(X).ravel())
#         tr_cost = cmodel.cost(machines, tr_pred, 0.5)
#         # Store the model weights
#         stored_weights[thr, c0_weight] = mdl.get_weights()
#         # Print info
#         if verbose > 0:
#             s = f'thr: {thr:.3f}, w0: {c0_weight:.3f}'
#             s += f', cost: {tr_cost}, time: {dur:.2f}'
#             print(s)
#         # Return the cost
#         return tr_cost

#     # Define the bouding box
#     box = {'thr': (0.0, 0.05), 'c0_weight': (1., 5.)}

#     # Start optimization
#     res = skopt.gp_minimize(f,
#             [box[k] for k in ('thr', 'c0_weight')],
#             acq_func='gp_hedge',
#             n_calls=n_iter,
#             n_random_starts=init_points,
#             noise=None,
#             random_state=seed,
#             verbose=False)

#     # Set the best weights
#     best_thr, best_c0_weight = res.x
#     mdl.set_weights(stored_weights[best_thr, best_c0_weight])
#     return res


def plot_ml_model(model):
    return keras.utils.plot_model(model, show_shapes=True,
            show_layer_names=True, rankdir='LR')


# ==============================================================================
# Data manipulation
# ==============================================================================

def partition_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def split_datasets_by_field(data, field, fraction, seed=None):
    # Seed the RNG
    np.random.seed(seed)
    # Obtain unique values of the field
    unq_vals = np.unique(data[field])
    # Shuffle
    np.random.shuffle(unq_vals)
    # Separate
    sep = int(len(unq_vals) * fraction)
    l1_vals = unq_vals[:sep]
    l1_vals = set(l1_vals)
    list1, list2 = [], []
    for val, gdata in data.groupby(field):
        if val in l1_vals:
            list1.append(gdata)
        else:
            list2.append(gdata)
    # Collate again
    data1 = pd.concat(list1)
    if len(list2) > 0:
        data2 = pd.concat(list2)
    else:
        data2 = pd.DataFrame(columns=data1.columns)
    return data1, data2


def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.tight_layout()



class RULCostModel:
    def __init__(self, maintenance_cost, safe_interval=0):
        self.maintenance_cost = maintenance_cost
        self.safe_interval = safe_interval

    def cost(self, machine, pred, thr, return_margin=False):
        # Merge machine and prediction data
        tmp = np.array([machine, pred]).T
        tmp = pd.DataFrame(data=tmp,
                           columns=['machine', 'pred'])
        # Cost computation
        cost = 0
        nfails = 0
        slack = 0
        for mcn, gtmp in tmp.groupby('machine'):
            idx = np.nonzero(gtmp['pred'].values < thr)[0]
            if len(idx) == 0:
                cost += self.maintenance_cost
                nfails += 1
            else:
                cost -= max(0, idx[0] - self.safe_interval)
                slack += len(gtmp) - idx[0]
        if not return_margin:
            return cost
        else:
            return cost, nfails, slack



def optimize_threshold(machine, pred, th_range, cmodel,
        plot=False, figsize=None):
    # Compute the optimal threshold
    costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
    opt_th = th_range[np.argmin(costs)]
    # Plot
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(th_range, costs)
        plt.xlabel('threshold')
        plt.ylabel('cost')
        plt.tight_layout()
    # Return the threshold
    return opt_th


def rul_cutoff_and_removal(data, cutoff_min, cutoff_max, seed=None):
    # Reseed the RNG
    np.random.seed(seed)
    # Loop over all machines
    data_by_m = partition_by_field(data, 'machine')
    for mcn, tmp in data_by_m.items():
        # Revemove final rows in each sequence
        cutoff = int(np.random.randint(cutoff_min, cutoff_max, 1))
        data_by_m[mcn] = tmp.iloc[:-cutoff]
    # Merge back
    res = pd.concat(data_by_m.values())
    # Delete RUL values
    res['rul'] = -1
    # Return results
    return res



class CstBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, in_cols, batch_size, seed=42):
        super(CstBatchGenerator).__init__()
        self.data = data
        self.in_cols = in_cols
        self.dpm = partition_by_field(data, 'machine')
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        # Build the first sequence of batches
        self.__build_batches()

    def __len__(self):
        return len(self.batches)

    # def __getitem__(self, index):
    #     idx = self.batches[index]
    #     mcn = self.machines[index]
    #     x = self.data[self.in_cols].loc[idx].values
    #     y = self.data['rul'].loc[idx].values
    #     return x, y


    def __getitem__(self, index):
        idx = self.batches[index]
        # mcn = self.machines[index]
        x = self.data[self.in_cols].loc[idx].values
        y = self.data['rul'].loc[idx].values
        flags = (y != -1)
        info = np.vstack((y, flags, idx)).T
        return x, info

    def on_epoch_end(self):
        self.__build_batches()

    def __build_batches(self):
        self.batches = []
        self.machines = []
        # Randomly sort the machines
        # self.rng.shuffle(mcns)
        # Loop over all machines
        mcns = list(self.dpm.keys())
        for mcn in mcns:
            # Obtain the list of indices
            index = self.dpm[mcn].index
            # Padding
            padsize = self.batch_size - (len(index) % self.batch_size)
            padding = self.rng.choice(index, padsize)
            idx = np.hstack((index, padding))
            # Shuffle
            self.rng.shuffle(idx)
            # Split into batches
            bt = idx.reshape(-1, self.batch_size)
            # Sort each batch individually
            bt = np.sort(bt, axis=1)
            # Store
            self.batches.append(bt)
            self.machines.append(np.repeat([mcn], len(bt)))
        # Concatenate all batches
        self.batches = np.vstack(self.batches)
        self.machines = np.hstack(self.machines)
        # Shuffle the batches
        bidx = np.arange(len(self.batches))
        self.rng.shuffle(bidx)
        self.batches = self.batches[bidx, :]
        self.machines = self.machines[bidx]



class CstRULRegressor(keras.Model):
    def __init__(self, rul_pred, alpha, beta, maxrul):
        super(CstRULRegressor, self).__init__()
        # Store the base RUL prediction model
        self.rul_pred = rul_pred
        # Weights
        self.alpha = alpha
        self.beta = beta
        self.maxrul = maxrul
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

        self.cnt = 0

    def call(self, data):
        return self.rul_pred(data)

    def train_step(self, data):
        x, info = data
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        with tf.GradientTape() as tape:
            # Obtain the predictions
            y_pred = self.rul_pred(x, training=True)
            # Compute the main loss
            mse = tf.math.reduce_mean(flags * tf.math.square(y_pred-y_true))
            # Compute the constraint regularization term
            delta_pred = y_pred[1:] - y_pred[:-1]
            delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
            deltadiff = delta_pred - delta_rul
            cst = tf.math.reduce_mean(tf.math.square(deltadiff))
            loss = self.alpha * mse + self.beta * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


def load_communities_data(data_folder, nan_discard_thr=0.05):
    # Read the raw data
    fname = os.path.join(data_folder, 'CommViolPredUnnormalizedData.csv')
    data = pd.read_csv(fname, sep=';', na_values='?')
    # Discard columns
    dcols = list(data.columns[-18:-2]) # directly crime related
    dcols = dcols + list(data.columns[7:12]) # race related
    dcols = dcols + ['nonViolPerPop']
    data = data.drop(columns=dcols)
    # Use relative values
    for aname in data.columns:
        if aname.startswith('pct'):
            data[aname] = data[aname] / 100
        elif aname in ('numForeignBorn', 'persEmergShelt',
                       'persHomeless', 'officDrugUnits',
                       'policCarsAvail', 'policOperBudget', 'houseVacant'):
            data[aname] = data[aname] / (data['pop'] / 100e3)
    # Remove redundant column (a relative columns is already there)
    data = data.drop(columns=['persUrban', 'numPolice',
                              'policeField', 'policeCalls', 'gangUnit'])
    # Discard columns with too many NaN values
    thr = nan_discard_thr * len(data)
    cols = data.columns[data.isnull().sum(axis=0) >= thr]
    cols = [c for c in cols if c != 'violentPerPop']
    data = data.drop(columns=cols)
    # Remove all NaN values
    data = data.dropna()
    # Shuffle
    rng = np.random.default_rng(42)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    return data.iloc[idx]


def plot_pred_by_protected(data, pred, protected, figsize=None):
    plt.figure(figsize=figsize)
    # Prepare the data for the boxplot
    x, lbls = [], []
    # Append the baseline
    pred = pred.ravel()
    x.append(pred)
    lbls.append('all')
    # Append the sub-datasets
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            x.append(pred[mask])
            lbls.append(f'{aname}={val}')
    plt.boxplot(x, labels=lbls)
    plt.tight_layout()


def DIDI_r(data, pred, protected):
    res = 0
    avg = np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            res += abs(avg - np.mean(pred[mask]))
    return res


class CstDIDIModel(keras.Model):
    def __init__(self, base_pred, attributes, protected, alpha, thr):
        super(CstDIDIModel, self).__init__()
        # Store the base predictor
        self.base_pred = base_pred
        # Weight and threshold
        self.alpha = alpha
        self.thr = thr
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def call(self, data):
        return self.base_pred(data)

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.base_pred(x, training=True)
            mse = self.compiled_loss(y_true, y_pred)
            # Compute the constraint regularization term
            ymean = tf.math.reduce_mean(y_pred)
            didi = 0
            for aidx, dom in self.protected.items():
                for val in dom:
                    mask = (x[:, aidx] == val)
                    didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
            cst = tf.math.maximum(0.0, didi - self.thr)
            loss = mse + self.alpha * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


class LagDualDIDIModel(keras.Model):
    def __init__(self, base_pred, attributes, protected, thr):
        super(LagDualDIDIModel, self).__init__()
        # Store the base predictor
        self.base_pred = base_pred
        # Weight and threshold
        self.alpha = tf.Variable(0., name='alpha')
        self.thr = thr
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def call(self, data):
        return self.base_pred(data)

    def __custom_loss(self, x, y_true, sign=1):
        y_pred = self.base_pred(x, training=True)
        # loss, mse, cst = self.__custom_loss(x, y_true, y_pred)
        mse = self.compiled_loss(y_true, y_pred)
        # Compute the constraint regularization term
        ymean = tf.math.reduce_mean(y_pred)
        didi = 0
        for aidx, dom in self.protected.items():
            for val in dom:
                mask = (x[:, aidx] == val)
                didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
        cst = tf.math.maximum(0.0, didi - self.thr)
        loss = mse + self.alpha * cst
        return sign*loss, mse, cst

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=1)

        # Separate training variables
        tr_vars = self.trainable_variables
        wgt_vars = tr_vars[:-1]
        mul_vars = tr_vars[-1:]

        # Update the network weights
        grads = tape.gradient(loss, wgt_vars)
        self.optimizer.apply_gradients(zip(grads, wgt_vars))

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=-1)

        grads = tape.gradient(loss, mul_vars)
        self.optimizer.apply_gradients(zip(grads, mul_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


def load_classification_dataset(csv_name, onehot_inputs=[]):
    # Load data
    data = pd.read_csv(csv_name)
    # Apply one-hot encoding
    if len(onehot_inputs) > 0:
        data = pd.get_dummies(data, columns=onehot_inputs)
    return data


class MLPLearner(object):

    def __init__(self, hidden, epochs=20, batch_size=32,
            epochs_fine_tuning=None, verbose=0):
        self.hidden = hidden
        self.epochs = epochs
        self.epochs_fine_tuning = epochs_fine_tuning
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.history = None
        self.wall_time = None

    def fit(self, X, y):
        # Build a model
        input_size = X.shape[1]
        output_size = y.shape[1]
        if self.epochs_fine_tuning is None or self.model is None:
            self.model = build_ml_model(input_size, output_size,
                    self.hidden, output_activation='softmax')
        # Train the model
        start = time.time()
        epochs_local = self.epochs if self.epochs_fine_tuning is None else self.epochs_fine_tuning
        self.history = train_ml_model(self.model, X, y, epochs=epochs_local,
                batch_size=self.batch_size, verbose=self.verbose, loss='categorical_crossentropy')
        self.wall_time = time.time() - start

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        y_idx = np.argmax(y_prob, axis=1)
        return y_idx


def avg_bal_deviation(y, bal_thr, nclasses):
    # Define the refence value for the balance constraint
    ref = len(y) / nclasses
    # Compute class counts
    val, cnt = np.unique(y, return_counts=True)
    # Compute the balance metric value
    # viol = np.maximum(0, np.abs(cnt - ref) / ref - bal_thr)
    viol = np.abs(cnt - ref) / ref
    return np.mean(viol)


def mt_learner(X, y, model, prob_output=False):
    # Train the model
    model.fit(X, y)
    # Return the prediction vector
    if not prob_output:
        return model.predict(X)
    else:
        return model.predict_proba(X)


# def mt_balance_master(X, y_true, y_pred, loss, bal_thr,
#         time_limit=None, verbose=0):
#     # Compute some useful parameters
#     ns = len(y_true)
#     nc = y_true.shape[1]

#     # Build a model
#     mdl = scip.Model('MT Master')
#     # Set a time limit (if needed)
#     if time_limit is not None:
#         mdl.setParam('limits/time', time_limit)
#     # Define verbosity
#     if verbose == 0:
#         mdl.hideOutput()

#     # Build target variables
#     z = {(i,j) : mdl.addVar(f'z_{i}', vtype='B') for i in range(ns) for j in range(nc)}
#     # Define the loss w.r.t. the original labels
#     loss_true = 0
#     # Unique class constraints
#     for i in range(ns):
#         mdl.addCons(sum(z[i, j] for j in range(nc)) == 1)

#     # Add the balance constraint
#     ref = ns / nc
#     for j in range(nc):
#         mdl.addCons(sum(z[i, j] for i in range(ns)) <= ref + ref * bal_thr)
#         mdl.addCons(sum(z[i, j] for i in range(ns)) >= ref - ref * bal_thr)

#     # Add the objective

#     # Solve
#     mdl.optimize()
#     status = mdl.getStatus()
#     assert status in ('optimal', 'feasible')
#     # Extract the target array
#     mt = np.zeros((ns, nc))
#     for i in range(ns):
#         for j in range(nc):
#             mt[i, j] = max(0, mdl.getVal(z[i, j]))
#     # Return the results
#     return mt


def mt_balance_master(y_true, y_pred, bal_thr, alpha=1,
        time_limit=None, mode='gradient'):
    assert mode in ('gradient', 'direct', 'projection', 'original')
    # Compute some useful parameters
    ns = len(y_true)
    nc = y_true.shape[1]

    # Build a model
    slv = pywraplp.Solver.CreateSolver('CBC')
    # Set a time limit (if needed)
    if time_limit is not None:
        slv.SetTimeLimit(1000 * time_limit)

    # Build target variables
    z = {(i,j) : slv.IntVar(0, 1, f'z[{i},{j}]') for i in range(ns) for j in range(nc)}
    # Unique class constraints
    for i in range(ns):
        slv.Add(sum(z[i, j] for j in range(nc)) == 1)

    # Add the balance constraint
    ref = ns / nc
    for j in range(nc):
        slv.Add(sum(z[i, j] for i in range(ns)) <= ref + ref * bal_thr)
        slv.Add(sum(z[i, j] for i in range(ns)) >= ref - ref * bal_thr)

    # Define a customized sign function
    def dsgn(yp, yt):
        d = yp - yt
        if d > 0: return 1
        elif d < 0: return -1
        else:
            if yp == 1: return -1 # works better for discrete predictions
            else: return 1 # works better for discrete predictions

    # Add the objective (new version)
    if mode == 'gradient':
        # Build the gradient-based part of the objective
        # - loss function: \sum_i \sum_j 0.5 * |y_i - \hat{y}_i|
        # - (sub)gradient: \sum_i \sum_j 0.5 * sgn(y_i - \hat{y}_i)
        # - (sub)gradient loss: \sum_i \sum_j 0.5 * sgn(y_i - \hat{y}_i) (z_i - y_i)
        loss_t = 0
        for i in range(ns):
            for j in range(nc):
                loss_t += 0.5 * dsgn(y_pred[i, j], y_true[i, j]) * (z[i, j] - y_pred[i, j])
    else:
        # Hamming distance w.r.t. true labels
        loss_t = 0
        for i in range(ns):
            for j in range(nc):
                loss_t += 0.5 * (y_true[i, j] * (1 - z[i, j]) + (1 - y_true[i, j]) * z[i, j])

    # Build the quadratic part of the objective
    # - loss function: \sum_i \sum_j (z_i - y_i)^2
    # - linearize loss function: \sum_i \sum_j z_i (1 - y_i)^2 + (1 - z_i) (0 - y_i)^2
    if mode != 'projection' and mode != 'original':
        loss_p = 0
        for i in range(ns):
            for j in range(nc):
                loss_p += z[i, j] * (1 - y_pred[i, j])**2 + (1 - z[i, j]) * (0 - y_pred[i, j])**2
    elif mode == 'original':
        loss_p = 0
        for i in range(ns):
            for j in range(nc):
                loss_p += 0.5 * (z[i, j] * np.abs(1 - y_pred[i, j]) + (1 - z[i, j]) * np.abs(0 - y_pred[i, j]))

    # Define the cost function
    if mode != 'projection':
        slv.Minimize(alpha * loss_t + loss_p)
    else:
        slv.Minimize(loss_t)

    # Solve
    status = slv.Solve()

    # Extract stats
    stats = {'opt': status == slv.OPTIMAL,
             'time': slv.WallTime() / 1000}

    # Extract the adjusted target vector
    sol = None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        # Extract the target array
        sol = np.zeros((ns, nc))
        for i in range(ns):
            for j in range(nc):
                sol[i, j] = max(0, z[i, j].solution_value())

    # Return the results
    return sol, stats


def mt_balance_stats(y_true_prob, y_pred_prob, bal_thr):
    # Obtain conventional classes
    y_true = np.argmax(y_true_prob, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    # Compute accuracy
    acc = metrics.accuracy_score(y_true, y_pred)
    # Compute balance violation
    bal = avg_bal_deviation(y_pred, bal_thr, y_pred_prob.shape[1])
    # Return the results
    return acc, bal


def mt_balance_stats_master(y_true_prob, y_pred_prob, z_prob):
    # Obtain conventional classes
    y_true = np.argmax(y_true_prob, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    z = np.argmax(z_prob, axis=1)
    # Compute accuracy
    acc = metrics.accuracy_score(y_true, z)
    # Compute mean square distance
    mse = metrics.mean_squared_error(y_true, z)
    # Return the results
    return acc, mse


def mt_balance(X, y, learner, bal_thr, alpha=1, max_iter=1,
        verbose=0, mode='gradient', master_tlim=None):
    assert mode in ('gradient', 'direct', 'projection', 'original')
    # Prepare result data structures
    history = {'learner_acc':[], 'learner_bal':[],
            'master_acc':[], 'm_l_dist':[],
            'learner_time':[], 'master_time':[]}
    def update_stats_learner(yp, ltime):
        acc, bal = mt_balance_stats(y, yp, bal_thr)
        history['learner_acc'].append(acc)
        history['learner_bal'].append(bal)
        history['learner_time'].append(ltime)
    def update_stats_master(yp, zp, mtime):
        acc, mse = mt_balance_stats_master(y, yp, zp)
        history['master_acc'].append(acc)
        history['m_l_dist'].append(mse)
        history['master_time'].append(mtime)
    # Pretraining
    if mode != 'projection':
        learner.fit(X, y)
        yp = learner.predict_proba(X)
        update_stats_learner(yp, learner.wall_time)
    else:
        yp = y.copy() # This is used only to compute the distance stat
    # Start the main iteration
    for k in range(max_iter):
        # Compute local alpha
        alpha_l = alpha / (k + 1)
        # Master step
        zp, stats = mt_balance_master(y, yp, bal_thr, alpha_l,
                mode=mode, time_limit=master_tlim)
        update_stats_master(yp, zp, stats['time'])
        # Learner step
        learner.fit(X, zp)
        yp = learner.predict_proba(X)
        update_stats_learner(yp, learner.wall_time)
        # Print some information
        if verbose > 0:
            lacc, lbal, ltime, macc, dist, mtime = \
                    history['learner_acc'][-1], \
                    history['learner_bal'][-1], \
                    history['learner_time'][-1], \
                    history['master_acc'][-1], \
                    history['m_l_dist'][-1], \
                    history['master_time'][-1]
            s = f'(#{k+1}) l-acc: {lacc:.2f}, l-bal: {lbal:.2f}, l-time: {ltime:.2f}s'
            s += f', m-acc: {macc:.2f}, l-m dist: {dist:.2f}, m-time: {mtime:.2f}s'
            print(s)
        if mode == 'projection':
            break
    # Return history
    return history


def mtx_function_plot(xm, ym, f_true=None, f_pred=None, figsize=None):
    fig = plt.figure(figsize=figsize)
    plt.scatter(xm, ym, color='tab:red', label='measures')
    span = xm[1] - xm[0]
    x = np.linspace(xm[0] - 0.5*span, xm[1] + 0.5*span)
    if f_true is not None:
        plt.plot(x, f_true(x), linestyle=':', color='tab:blue', label='true function')
    if f_pred is not None:
        plt.plot(x, f_pred(x), color='tab:orange', label='estimated function', linestyle=':')
        plt.scatter(xm, f_pred(xm), color='tab:orange', label='predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(linestyle=':')
    plt.axis('equal')
    plt.legend()

    
def mtx_output_plot(xm, ym, yp, f_bound=None, yf=None, plot_bias=False,
        figsize=None, ylim=None):
    fig = plt.figure(figsize=figsize)
    y0 = np.linspace(ym[0] - 0.02, ym[0] + 0.22)
    y1 = xm[1]**(np.log(y0) / np.log(xm[0]))
    plt.gca().set_facecolor((0.95, 0.95, 0.95))
    plt.axvspan(y0[0], y0[-1], color='white')
    if plot_bias:
        plt.plot(y0, y1, color='tab:orange', label='ML model bias')
    plt.scatter(ym[0], ym[1], color='tab:red', label='measured values', zorder=2)
    if yp is not None:
        yp = np.array(yp).reshape(-1, 2)
        plt.scatter(yp[:, 0], yp[:, 1], color='tab:orange', label='predictions', zorder=2)
    if f_bound:
        y1_bound = f_bound(y0)
        ylim = max(plt.ylim()[0], ylim[0]), min(plt.ylim()[1], ylim[1])
        plt.fill_between(y0, ylim[0], y1_bound, zorder=2, alpha=0.2, label='feasible output')
    if yf is not None:
        yf = np.array(yf).reshape(-1, 2)
        plt.scatter(yf[:, 0], yf[:, 1], color='tab:blue', label='adjusted target', zorder=2)
    if yf is not None and yp is not None:
        tmp = np.empty((len(yp)+len(yf), yp.shape[1]))
        tmp[0::2, :] = yp
        tmp[1::2, :] = yf
        plt.plot(tmp[:, 0], tmp[:, 1], linestyle=':', color='0.5', zorder=1)
    plt.xlabel('y0')
    plt.ylabel('y1')
    plt.grid(linestyle=':')
    plt.axis('equal')
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)

def mtx_learner_step(xm, ym):
    f = lambda x, a: x**a
    p = curve_fit(f, xm, ym)
    a_opt = p[0][0]
    return lambda x: x**a_opt


def mtx_master_step_alpha(ym, yp, alpha=0.1):
    cst = LinearConstraint([[1.5, -1]], 0, np.inf)
    obj = lambda y: alpha*2*np.dot((yp - ym), (y - yp)) + np.square(y - yp).sum()
    res = minimize(obj, yp, constraints=[cst])
    return res.x


def mtx_moving_target_alpha(xm, ym, n, alpha=0.1):
    ypl = []
    yfl = []

    f_pred = mtx_learner_step(xm, ym)
    yp = f_pred(xm)
    ypl.append(yp)
    for i in range(n):
        # Master step
        yf = mtx_master_step_alpha(ym, yp, alpha=alpha/(i+1))
        yfl.append(yf)
        # Learner stpe
        f_pred = mtx_learner_step(xm, yf)
        yp = f_pred(xm)
        ypl.append(yp)
    return ypl, yfl, f_pred

