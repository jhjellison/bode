import numpy as np
import seaborn as sns
sns.set_style("white")    #Changes the seaborn plotting sytle
sns.set_context("paper")
import matplotlib.pyplot as plt
plt.switch_backend('agg') #Changes matplotlib image generating engine to one optimised for raster graphics (i.e. png files)
import os
import shutil
import GPy
import pickle
from bode import KLSampler, GammaPrior, ExponentialPrior, JeffreysPrior
from pyDOE import lhs
from scipy.stats import norm
#from scipy.stats import uniform #Used in some quadrature points generating functions - not currently needed

    ####################################
    ####################################

class Ex1Func(object):
    # Initialize the Ex1Func object with a sigma function
    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    # Make instances of Ex1Func callable.  Creates a not-trivial, non-linear function with noise
    def __call__(self, x):
        #Creates function
        x = 6 * x
        function_value = (4 * (1. - np.sin(x[0] + 8 * np.exp(x[0] - 7.))) -10.) / 5
        #Adds noise
        noisy_value = function_value + self.sigma(x[0]) * np.random.randn()
        return noisy_value

    ####################################
    ####################################

if __name__=='__main__':
    # Set the random seed for reproducibility
    np.random.seed(1333)
    
    # Define initial parameters
    n = 3  # Number of initial samples of the data
    n_true = 100000  # Number of samples to estimate the true mean - only used for comparison not in the sampler/estimation
    dim = 1  # Dimensionality of the input space
    noise = 0.05  # Noise level for the objective function
    noise_true = 0  # Noise level for the true objective function
    # Define parameters for quadrature points and MCMC
    num_quad_points = 500
    quad_points = lhs(dim, num_quad_points)
    quad_points_weight = np.ones(num_quad_points)
    # Define hyperparameters for the MCMC sampler
    mu = 0.5
    sigma = 0.2
    num_it = 3  #Number of interations i.e. how many additional points to create
    #Define parameters for plotting
    size_plots = 10000 #

    #Note one may also wish to change the parameters in the kls which is passed to the sampler

    # Additional options for generating quadrature points and their weights
	# quad_points = uniform.rvs(0, 1, size=num_quad_points)
	# quad_points_weight = uniform.pdf(quad_points)
	# quad_points = norm.rvs(mu, sigma, size=num_quad_points)
	# quad_points_weight = norm.pdf(quad_points, mu, sigma)
	# quad_points = np.linspace(0.01, .99, num_quad_points)[:, None] 			# Linearly space points
	# quad_points_weight = 1. / np.sqrt(1. - (quad_points[:, 0] ** 2)) 	# Right side heavy
	# quad_points_weight = 1./ np.sqrt(quad_points[:, 0])  # Left side heavy
	# quad_points_weight = 1./np.sqrt(abs(quad_points-0.5)) # Middle heavy
     
    ####################################
    ####################################
    
    # Define the noise functions using lambda expressions
    sigma = eval('lambda x: ' + str(noise))
    sigma_true = eval('lambda x: ' + str(noise_true))
    
    # Create instances of Ex1Func with the specified noise levels
    objective_true = Ex1Func(sigma=sigma_true)
    objective = Ex1Func(sigma=sigma)
    
    # Generate initial sample points using Latin Hypercube Sampling (LHS)
    X_init = lhs(dim, samples=n, criterion='center')
    Y_init = np.array([objective(x) for x in X_init])[:, None]
    
    # Generate true sample points for estimating the true mean
    X_true = lhs(dim, samples=n_true, criterion='center')
    Y_true = np.array([objective_true(x) for x in X_true])[:, None]
    
    # Compute the true mean of the objective function - only used for comparison not in the sampler/estimation
    true_mean = np.mean(Y_true)
    print(true_mean)
    print('true E[f(x)]: ', true_mean)
    
    # Define output directory and create it
    out_dir = 'mcmc_ex1_n={0:d}_it={1:d}'.format(n, num_it)
    if os.path.isdir(out_dir):
        #If the directory exists delete the entire directory
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    # Hypothetical point for the KLSampler (not used in current code)
    x_hyp = np.array([[.6]])
    
    # Initialize the KLSampler
    kls = KLSampler(X_init, Y_init, x_hyp=False,        # Pass inital X and Y values
        model_kern=GPy.kern.RBF,                        # Define kernel for GP
        bounds=[(0,1)] * X_init.shape[1],               # Bounds for the input space, here it is [0,1] for each dimension
        obj_func=objective,                             # Objective function to be optimized
        true_func=objective_true,                       # True objective function (without noise)
        noisy=True,                                    # Specifies if the objective function is noisy
        nugget=1e-3,
        lengthscale=0.2,                                # Lengthscale parameter for the RBF kerne
        variance=1.,                                    # Variance parameter for the RBF kernel
        kld_tol=1e-6,
        func_name=os.path.join(out_dir,'ex1'),          # Output directory and base name for saved files
        energy=0.95,
        num_quad_points=num_quad_points,
        quad_points=quad_points,
        quad_points_weight=quad_points_weight,
        max_it=num_it,                                  # Number of iteractions unless other stopping points are reached
        per_sampled=20,
        mcmc_model=True,
        mcmc_chains=6,
        mcmc_model_avg = 60,	#should be a multiple of number of chains
        mcmc_steps=1000,
        mcmc_burn=200,
        mcmc_thin=20,
        mcmc_parallel=8,
        ego_iter=20,
        initialize_from_prior=False,                    # Use pirors
        variance_prior=GammaPrior(a=1, scale=2),        # Adds piror conditioning to the variance
        lengthscale_prior=ExponentialPrior(scale=0.7),  # Adds piror conditioning to the lengthscale
        noise_prior=JeffreysPrior())                    # Adds piror conditioning to the noise
    
    # Optimize the KLSampler
    X, Y, X_u, kld, X_design, mu_qoi, sigma_qoi, comp_log = kls.optimize(num_designs=1000, verbose=1, plots=1, comp=True, comp_plots=False)
    
    # Save the results
    np.save(os.path.join(out_dir,'X.npy'), X)
    np.save(os.path.join(out_dir,'Y.npy'), Y)
    np.save(os.path.join(out_dir,'X_u.npy'), X_u)
    np.save(os.path.join(out_dir,'kld.npy'), kld)
    np.save(os.path.join(out_dir,'mu_qoi.npy'), mu_qoi)
    np.save(os.path.join(out_dir,'sigma_qoi.npy'), sigma_qoi)
    with open(os.path.join(out_dir, "comp.pkl"), "wb") as f:
        pickle.dump(comp_log, f)
        
    # Compute the maximum KLD values for each iteration
    kld_max = np.ndarray(kld.shape[0])
    for i in range(kld.shape[0]):
        kld_max[i] = max(kld[i, :])
     
    # Generate random samples for the distributions
    x = np.ndarray((size_plots, len(mu_qoi)))
    x_us = np.ndarray((size_plots, len(mu_qoi)))
    pos = np.arange(n, n + len(mu_qoi))
    
    ####################################
    ####################################   
    
    # Plot the relative maximum KLD values
    plt.plot(np.arange(len(kld_max)), kld_max / max(kld_max), color=sns.color_palette()[1])
    plt.xlabel('iterations', fontsize=16)
    plt.ylabel('relative maximum $G(x)$', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(out_dir,'ekld.png'), dpi=(900))
    plt.clf()
    
    # Plot the true mean value to be combined with box plot
    plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the $Q$', linewidth=4)
    for i in range(len(mu_qoi)):
        x[:, i] = norm.rvs(loc=mu_qoi[i], scale=sigma_qoi[i] ** .5, size=size_plots)
        x_us[:, i] = norm.rvs(loc=comp_log[0][i], scale=comp_log[1][i] ** .5, size=size_plots)
    # Box plot for the EKLD results
    bp_ekld = plt.boxplot(x, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
    plt.xlabel('no. of samples', fontsize=16)
    plt.ylabel('$Q$', fontsize=16)
    plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(out_dir, 'box.png'), dpi=(900))
    plt.clf()
    
    # Distribution plots for the initial and final distributions of Q
    sns.distplot(norm.rvs(loc=mu_qoi[0], scale=sigma_qoi[0] ** .5, size=size_plots), color=sns.color_palette()[1], label='initial distribution of $Q$', norm_hist=True)
    sns.distplot(norm.rvs(loc=mu_qoi[-1], scale=sigma_qoi[-1] ** .5, size=size_plots), hist=True, color=sns.color_palette()[0], label='final distribution of $Q$', norm_hist=True)
    plt.scatter(true_mean, 0, c=sns.color_palette()[2], label='true value of the $Q$')
    plt.legend(fontsize=12)
    plt.xlabel('$Q$', fontsize=16)
    plt.ylabel('$p(Q|\mathrm{data})$', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(out_dir, 'dist.png'), dpi=(900))
    plt.clf()
	
    # Comparison plot of the true mean, EKLD, and uncertainty sampling
    plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the $Q$', linewidth=4)
    plt.plot(pos, mu_qoi, '-o',  color=sns.color_palette()[1], label='EKLD', markersize=10)
    plt.plot(pos, comp_log[0], '-*',  color=sns.color_palette()[2], label='uncertainty sampling', markersize=10)
    plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('no. of samples', fontsize=16)
    plt.ylabel('$\mathbb{E}[Q]$', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(out_dir, 'comparison.png'), dpi=(900))
