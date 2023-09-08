import numpy as np
import torch
from torch.autograd import grad
import torch.distributions as D
from torch_scatter import scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Ensemble(torch.nn.Module):
    """
    A wrapper class for an Uncertainty-Quantification Model
    Args:
        - model_name: "dropout_dimenet++" or "evidential_dimenet++"
        - model: trained model
    """
    def __init__(self, model_name, models):
        super().__init__()
        self.model_name = model_name
        self.models = models
        self.n_models = len(models)
        
    def forward(self, inputs):
        """
        Args:
            inputs: input batch containing inputs.z, inputs,pos, inputs.batch
        Returns:
            Dict consisting of:
                - energy: energy prediction
                - forces: forces prediction
                - uncertainty_1: energy std for dropout model / epistemic for evidential model
                - uncertainty_2: forces cov det for dropout model / aleatoric for evidential model
        """
        energy_samples, forces_samples = self.predict(inputs)
        with torch.no_grad():
            energy, forces = energy_samples.mean(0).to(device), forces_samples.mean(0).to(device)
            energy_uncertainty = energy_samples.std(0).squeeze().to(device)
            det, trace, largest_eig = self.compute_forces_uncertainty_molecule(forces_samples, inputs)
            
            try:             
                energy_distribution = D.normal.Normal(energy, energy_uncertainty + 10e-6)
            except:
                print('not working')
                energy_distribution = D.normal.Normal(energy, energy_uncertainty)
            return {
                "energy": energy, 
                "forces": forces, 
                "energy_uncertainty": energy_uncertainty, 
                "det": det.to(device), 
                "trace": trace.to(device), 
                "largest_eig": largest_eig.to(device), 
                "energy_distribution": energy_distribution
            }
            
    def predict(self, inputs):
        energy_samples = []
        forces_samples = []
        orig_data = inputs.clone()
        for model in self.models:
            inputs = orig_data.clone()
            inputs.pos.requires_grad = True
            energy = model(inputs)
            forces = - grad(
                outputs=energy.sum(),
                inputs=inputs.pos,
                create_graph=True,
                retain_graph=True
            )[0]
            energy_samples.append(energy)
            forces_samples.append(forces)
            inputs.pos.requires_grad = False
            
        energy_samples = torch.stack(energy_samples).detach().cpu()
        forces_samples = torch.stack(forces_samples).detach().cpu()
        return (energy_samples, forces_samples)
    
    def compute_forces_uncertainty_atomwise(self, forces_samples, inputs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        det = torch.zeros(forces_samples.shape[1], device=device) # Number of atoms
        trace = torch.zeros(forces_samples.shape[1], device=device)
        largest_eig = torch.zeros(forces_samples.shape[1], device=device)
        for atom_idx in range(forces_samples.shape[1]):
            atom_force = forces_samples[:, atom_idx, :]
            atom_cov = torch.tensor(np.cov(atom_force.double().T.cpu().numpy())).to(device)
            det[atom_idx] = torch.det(atom_cov)
            trace[atom_idx] = torch.trace(atom_cov)
            eig_vals = torch.linalg.eigvalsh(atom_cov)
            largest_eig[atom_idx] = eig_vals[-1]

        det = scatter(det, inputs.batch, dim=0, reduce="mean") # Average over atoms for each molecule
        trace = scatter(trace, inputs.batch, dim=0, reduce="mean")
        largest_eig = scatter(largest_eig, inputs.batch, dim=0, reduce="mean")
        return det, trace, largest_eig
    
    def compute_forces_uncertainty_molecule(self, forces_samples, inputs):
        # forces_samples has shape (n_mc_dropout_runs, N_atoms, 3) -> we make it (n_mc_dropout_runs, 3*N_atoms)
        forces_samples = torch.reshape(forces_samples, (forces_samples.shape[0], -1))
        cov_mat = torch.tensor(np.cov(forces_samples.double().T.cpu().numpy())).to(device)
        det = torch.det(cov_mat)
        trace = torch.trace(cov_mat)
        eig_vals = torch.linalg.eigvalsh(cov_mat)
        largest_eig = eig_vals[-1]

        return det, trace, largest_eig
    
    def eval(self):
        for model in self.models:
            model.eval()
        return self

    def train(self):
        for model in self.models:
            model.train()
        return self