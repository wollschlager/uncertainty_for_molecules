import numpy as np
import torch
from torch_scatter import scatter
import torch.distributions as D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MCD(torch.nn.Module):
    """
    A wrapper class for an MCD Model for inference
    Args:
        - model_name: "name of the backbone model"
        - model: trained model
    """
    def __init__(self, model_name, model, n_mc_dropout_runs=5):
        super().__init__()
        self.backbone_name = model_name
        self.model = model
        self.n_mc_dropout_runs = n_mc_dropout_runs
        
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
        energy_samples, forces_samples = self.predict_MC_Dropout(inputs, n_mc_dropout_runs=self.n_mc_dropout_runs)
        with torch.no_grad():
            energy, forces = energy_samples.mean(0).to(device), forces_samples.mean(0).to(device)
            energy_uncertainty = energy_samples.std(0).squeeze().to(device)
            det, trace, largest_eig = self.compute_forces_uncertainty_molecule(forces_samples, inputs)
                            
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

            
    def predict_MC_Dropout(self, inputs, n_mc_dropout_runs=5):
        energy_samples = []
        forces_samples = []
        for _ in range(n_mc_dropout_runs):
            energy, forces = self.model(inputs)
            energy_samples.append(energy)
            forces_samples.append(forces)
            
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