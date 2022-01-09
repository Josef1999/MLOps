
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class PytorchModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    class DaGMM(nn.Module):
        """Residual Block."""
        def __init__(self, n_gmm = 2, latent_dim=3):
            super(PytorchModel.DaGMM, self).__init__()

            layers = []
            layers += [nn.Linear(118,60)]
            layers += [nn.Tanh()]        
            layers += [nn.Linear(60,30)]
            layers += [nn.Tanh()]        
            layers += [nn.Linear(30,10)]
            layers += [nn.Tanh()]        
            layers += [nn.Linear(10,1)]

            self.encoder = nn.Sequential(*layers)


            layers = []
            layers += [nn.Linear(1,10)]
            layers += [nn.Tanh()]        
            layers += [nn.Linear(10,30)]
            layers += [nn.Tanh()]        
            layers += [nn.Linear(30,60)]
            layers += [nn.Tanh()]        
            layers += [nn.Linear(60,118)]

            self.decoder = nn.Sequential(*layers)

            layers = []
            layers += [nn.Linear(latent_dim,10)]
            layers += [nn.Tanh()]        
            layers += [nn.Dropout(p=0.5)]        
            layers += [nn.Linear(10,n_gmm)]
            layers += [nn.Softmax(dim=1)]


            self.estimation = nn.Sequential(*layers)

            self.register_buffer("phi", torch.zeros(n_gmm))
            self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
            self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))

        def relative_euclidean_distance(self, a, b):
            return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

        def to_var(self, x):
            if torch.cuda.is_available():
                x = x.cuda()
            return Variable(x)
            
        def forward(self, x):

            enc = self.encoder(x)

            dec = self.decoder(enc)

            rec_cosine = F.cosine_similarity(x, dec, dim=1)
            rec_euclidean = self.relative_euclidean_distance(x, dec)

            z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

            gamma = self.estimation(z)

            return enc, dec, z, gamma

        def compute_gmm_params(self, z, gamma):
            N = gamma.size(0)
            # K
            sum_gamma = torch.sum(gamma, dim=0)

            # K
            phi = (sum_gamma / N)

            self.phi = phi.data

    
            # K x D
            mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
            self.mu = mu.data
            # z = N x D
            # mu = K x D
            # gamma N x K

            # z_mu = N x K x D
            z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

            # z_mu_outer = N x K x D x D
            z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

            # K x D x D
            cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
            self.cov = cov.data

            return phi, mu, cov
            
        def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
            if phi is None:
                phi = self.to_var(self.phi)
            if mu is None:
                mu = self.to_var(self.mu)
            if cov is None:
                cov = self.to_var(self.cov)

            k, D, _ = cov.size()

            z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

            cov_inverse = []
            det_cov = []
            cov_diag = 0
            eps = 1e-12
            for i in range(k):
                # K x D x D
                cov_k = cov[i] + self.to_var(torch.eye(D)*eps)
                cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

                #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
                det_cov.append((torch.cholesky(cov_k * (2*np.pi), False).diag().prod()).unsqueeze(0))
                #det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
                cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

            # K x D x D
            cov_inverse = torch.cat(cov_inverse, dim=0)
            # K
            if torch.cuda.is_available():
                det_cov = torch.cat(det_cov).cuda()
            else:
                det_cov = torch.cat(det_cov)
            #det_cov = self.to_var(torch.from_numpy(np.float32(np.array(det_cov))))

            # N x K
            exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
            # for stability (logsumexp)
            max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

            exp_term = torch.exp(exp_term_tmp - max_val)

            # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
            sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
            # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


            if size_average:
                sample_energy = torch.mean(sample_energy)

            return sample_energy, cov_diag


        def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

            recon_error = torch.mean((x - x_hat) ** 2)

            phi, mu, cov = self.compute_gmm_params(z, gamma)

            sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

            loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

            return loss, sample_energy, recon_error, cov_diag

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        model_name = "model.pth"
        
        self.clf = PytorchModel.DaGMM(4)
        self.clf.load_state_dict(torch.load(model_name, map_location=torch.device('cpu') ))
        self.clf.eval()

    def predict(self, X, features_names):
        """
        Return a prediction.
        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        enc, dec, z, gamma = self.clf(Variable((torch.tensor(X).to(torch.float32))))
        
        test_energy = []
        sample_energy, cov_diag =  self.clf.compute_energy(z, size_average=False)
        test_energy.append(sample_energy.data.cpu().numpy())
        test_energy = np.concatenate(test_energy,axis=0)
        thresh = np.percentile(test_energy, 100 - 20)

        result = (test_energy > thresh).astype(int)
        return result

if __name__ == '__main__':
    #测试模型导入正确性
    c = PytorchModel()
    element = np.array([0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 2.61041764e-07, 1.05713002e-03, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.56555773e-02,
 1.56555773e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 3.14960630e-02, 3.14960630e-02, 1.00000000e+00, 0.00000000e+00,
 1.10000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00])

    res = c.predict(np.array([element]),'whatever')
    print(res)