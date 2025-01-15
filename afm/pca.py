import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm

class PCA_fault_detection():


    def __init__ (self, n_component: int):
        """
        Constructor of the class
        """
        self.n_c = int(n_component)
   
 
    def fit(self,data:pd.DataFrame, alpha:float = 0.01, plot:bool = True):
        """"
        Method for training the detection model, setting detection limit and plotting the explained variance 
        Args :
            data : Normal operating condition data
            alpha : nominal risk of false alarms
            plot : plotting the explained variance
        """             
        # Data standardisation
        sc = StandardScaler()
        data_scaled = sc.fit_transform(data)
        self.mu_train = sc.mean_
        self.std_train = sc.scale_
        
        # Training the PCA
        pca = PCA()
        pca.fit(data_scaled)
        
        # Retrieving eigenvectors and eigenvalues
        self.L = pca.explained_variance_
        self.P = pca.components_.T
        # Variance explained by each eigenvalue
        fv = self.L/np.sum(self.L)
        
        # Cumulative variance explained by all eigenvalues
        fva = np.cumsum(self.L)/sum(self.L)

        # Setting the detection threshold based on Jackson and Mudholkar (1979)

        theta = [np.sum(self.L[self.n_c:]**(i)) for i in (1,2,3)]
        ho = 1-((2*theta[0]*theta[2])/(3*(theta[1]**2)))
        self.conf_Q = 1-alpha
        nalpha = norm.ppf(self.conf_Q)
        self.Q_lim = (theta[0]*(((nalpha*np.sqrt(2*theta[1]*ho**2))/theta[0])+1+
                                ((theta[1]*ho*(ho-1))/theta[0]**2))**(1/ho))
        
        # plotting the explained variance
        if plot:
            fig, ax = plt.subplots()
            ax.bar(np.arange(1,len(fv)+1),fv)
            ax.plot(np.arange(1,len(fv)+1),fva, color='orange',label='Cumulative explained variance')
            # Annotation for retained components
            plt.axvline(self.n_c, color="red", linestyle="--", label=f"{self.n_c} components retained")
            plt.text(
                self.n_c + 0.3, 0.5,
                f"{fva[self.n_c - 1]:.2%} explained variance",
                color="red", fontsize=10, verticalalignment="center"
            )
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Proportion of Explained Variance')
            ax.set_title('Explained Variance by Retained Components')
            
    def calculate_spe(self, X):
        '''
        Method that calculates the Square Prediction Error (SPE) of given data X
        '''
        # Standardisation of the data
        X = np.array((X-self.mu_train)/self.std_train)

        # Calculate the SPE or Q-statistics
        e = X - X@self.P[:,:self.n_c]@self.P[:,:self.n_c].T
        Q  = np.array([e[i,:]@e[i,:].T for i in range(X.shape[0])])
        return Q
