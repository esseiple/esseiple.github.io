import torch

class LinearModel:

    def __init__(self):
        self.w = None 
        self.pastw = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = (torch.rand(X.size()[1])-0.5) / X.size()[1]

        return torch.matmul(X, self.w)


    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """

        s = self.score(X)
        y_hat = 1.0*(s >= 0)

        return y_hat

class LogisticRegression(LinearModel):

    def loss(self, X, y, epsilon=1e-8):
        """
        Compute the logistic loss.
        """
        scores = self.score(X)
        sig_scores = torch.sigmoid(scores)
        ## adding epsilon values to prevent trying to take log of zero which gives me a loss of NaN
        loss = -y * torch.log(sig_scores + epsilon) - (1 - y) * torch.log(1 - sig_scores + epsilon)
        return torch.mean(loss)


    def grad(self, X, y):
        scores = self.score(X)
        sigscored = torch.sigmoid(scores)
        grad = (sigscored - y)[:, None] * X
        return torch.mean(grad, dim=0) 

    def hessian(self, X):
        """
        Compute the Hessian matrix.
        """
        scores = self.score(X)
        sig_scores = torch.sigmoid(scores)
        diag = sig_scores * (1 - sig_scores)
        d_mat = torch.diag(diag)
        hessian = X.T@d_mat@X
        return hessian


class NewtonOptimizer:
    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, a):
        """
        Compute one step of the Newton's method update using the feature matrix X 
        and target vector y, and the learning parameter alpha.
        """
        gradient = self.model.grad(X, y)
        hessian = self.model.hessian(X)
        self.model.w = self.model.w - (a * torch.linalg.inv(hessian))@gradient
        
        return 


class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, a, b):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y, and the learning parameters alpha and beta.
        """
        # calculate gradient and loss
        grad = self.model.grad(X, y)
        loss = self.model.loss(X, y)

        # store current weights
        weight_curr = self.model.w.clone()

        # if the model has no past weights
        if self.model.pastw == None:
            # set them according to LR equations
            self.model.w -= a*grad
        else:
            # store past weights
            weight_naught = self.model.pastw.clone()
            # update current weights based on LR equation (taking into account alpha, beta, past and current weights)
            self.model.w += (-1 * a * grad) + b * (weight_curr - weight_naught)

        # update past weights to current weights (before they were updated)
        self.model.pastw = weight_curr.clone()
            
        return loss
