import numpy as np
import torch
from ue4mol.models.backbones import DimeNetPPDropout, SchNetDropout, DimeNetPPDropoutMulti
from ue4mol.models.gmdn import GMDN
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from functools import partial
from gpytorch.kernels import RBFKernel, MaternKernel, ProductKernel, SpectralMixtureKernel
from tqdm import tqdm
from sklearn.cluster import KMeans
from enum import unique, Enum


@unique
class Models(str, Enum):
    DIMENET = 'dimenet++'
    SCHNET = 'schnet'
    NEQUIP = 'nequip'
    PAINN = 'painn'

def calc_inducing_points(model, train_loader, type, n_inducing_points, per_cluster_lengthscale=True):
    print("inducing point init: ", type)
    embs = []
    with torch.no_grad():
        energies = []
        model = model.cpu()
        model = model.cuda()
        for batch in tqdm(train_loader):
            batch = batch.cuda()
            output = model(batch.cuda())
            embs.append(output.cpu())
            energies.append(batch.energy.cpu())
        embs = torch.cat(embs).cpu()
        if type == "mult-gp":
            energies = []
            per_atom_emb = {}
            max_atoms = 0
            for batch in tqdm(train_loader):
                embedding = model(batch.cuda())
                energies.append(batch.energy.cpu())
                for i in range(max(batch.batch) + 1):
                    current = embedding[batch.batch == i]
                    max_atoms = max(max_atoms, len(current))
                    for j in range(max_atoms):
                        if j in per_atom_emb:
                            per_atom_emb[j].append(current[j:j+1, :])
                        else:
                            per_atom_emb[j] = [current[j:j+1, :]]
                            
            inducing_points = []
            lengthscales = []
            num_per_cluster = n_inducing_points // max_atoms
            for k in per_atom_emb:
                embs = torch.cat(per_atom_emb[k])
                mask = np.random.choice(len(embs), size=num_per_cluster)
                inducing_points.append(embs[mask])
                if per_cluster_lengthscale:
                    try: 
                        lengthscales.append(torch.pdist(embs.cpu()).mean())
                    except:
                        lengthscales.append(torch.pdist(embs.cpu()[torch.randint(0, len(embs), (10000,))]).mean())
                        
            inducing_points = torch.cat(inducing_points)
            
            if per_cluster_lengthscale:
                # calculate lengthscale only within each cluster
                initial_lengthscale = torch.mean(torch.tensor(lengthscales))
            else:
                # calculate lengthscales also between the clusters
                initial_lengthscale = torch.pdist(inducing_points)
        elif type == "k-means" or type == "kmeans":
            data = embs.numpy()
            kmeans = KMeans(n_clusters=n_inducing_points).fit(data)
            inducing_points = torch.tensor(kmeans.cluster_centers_)
            try:
                initial_lengthscale = torch.pdist(embs.cpu()).mean()
            except:
                initial_lengthscale = torch.pdist(embs.cpu()[torch.randint(0, len(embs), (50000,))]).mean()
        elif type == "first":
            inducing_points = torch.cat(embs)[:n_inducing_points]
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        elif type == "random":
            inducing_points = torch.randn((n_inducing_points, embs.shape[1]))
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        else:
            raise NotImplementedError
        energies = torch.cat(energies)[:n_inducing_points] # energys dont make sense for kmeans
    return inducing_points, energies, initial_lengthscale
        
        
def load_localized_backbone(model_name, model_params, pretrained):
    model = init_local_model(model_name, model_params)
    if pretrained is not None:
        model.load(pretrained)
    return model


def load_global_backbone(model_name, model_params, pretrained):
    model = init_global_model(model_name, model_params)
    if pretrained is not None:
        model.load(pretrained)
    return model


def load_fixed_fe(use_pretrained=True):
    encoder_name = "dimenet"
    encoder_params = {
        'hidden_channels': 128,
        'num_blocks': 6,
        'num_bilinear': 8,
        'num_spherical': 7,
        'num_radial': 6,
        'cutoff': 5.0,
        'envelope_exponent': 5,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_output_layers': 3,
        'out_channels': 1
    }
    if use_pretrained:
        pretrained = '/nfs/homedirs/wollschl/staff/uncertainty-molecules/models/dimenet_pretrained_U0'
    else:
        pretrained = None
    return load_global_backbone(encoder_name, encoder_params, pretrained)

def init_local_model(model_name, model_params):
    if model_name == Models.DIMENET:
        return DimeNetPPDropoutMulti(dimenet_pp_params=model_params)
    elif model_name == Models.SCHNET:
        return SchNetMultDropout(params=model_params)
    elif model_name == Models.NEQUIP:
        return NequIPMulti(params=model_params)
    else:
        print(f'Model {model_name} is not implemented')
        raise NotImplementedError


def init_global_model(model_name, model_params):
    if model_name == Models.DIMENET:
        return DimeNetPPDropout(**model_params)
    elif model_name == Models.SCHNET:
        return SchNetDropout(**model_params)
    elif model_name == Models.PAINN:
        return PaiNN(params=model_params)
    elif model_name == Models.NEQUIP:
        return NequIP(**model_params)
    elif model_name == "gmdn": 
        if 'encoder_params' in model_params:
           encoder_params = model_params['encoder_params'] 
        else:
            encoder_params = {}
        encoder = init_global_model(model_params['encoder_name'], model_params=encoder_params)
        encoder.to_encoder()
        
        return GMDN(**model_params, encoder=encoder)
    else:
        print(f'Model {model_name} is not implemented')
        raise NotImplementedError


def init_kernel(kernel_name, kernel_params=None):
    if kernel_params is not None:
        if "batch_shape" in kernel_params:
            kernel_params = kernel_params.copy()
            kernel_params["batch_shape"] = torch.Size([kernel_params["batch_shape"]])
    if kernel_name == 'rbf':
        kernel = RBFKernel(**kernel_params)
    elif kernel_name == 'matern':
        kernel = MaternKernel(**kernel_params)
    elif kernel_name == 'product_explinear_rbf':
        exp_linear = ExponentialLinear(**kernel_params['linear'])
        rbf = RBFKernel(**kernel_params['rbf'])
        kernel = ProductKernel(exp_linear, rbf)
    elif kernel_name == 'spectral_mixture':
        kernel = SpectralMixtureKernel(**kernel_params)
    else:
        raise NotImplementedError(f"kernel {kernel_name} is not implemented")
    return kernel


def wrap_prediction_model(type, model, loss_name, num_data, beta, rho_force, model_class='UQ'):
    likelihood = GaussianLikelihood()
    if loss_name is None or loss_name == 'elbo':
        loss_fn = partial(
            neg_var_elbo,
             elbo=VariationalELBO(likelihood, model.gp, num_data=num_data, beta=beta)
        )
    elif type == 'energy' and loss_name == 'mae': 
        loss_fn = torch_mae
    elif type == 'energy' and loss_name == 'torch_mae':
        loss_fn = torch_mae
    elif type == 'force' and loss_name == 'energy_mae_force_mae':
        loss_fn = partial(
            force_loss, 
            rho_force=rho_force,
            energy_loss_type="mae",
            force_loss_type="mae"
        )
    elif type == 'force' and loss_name == 'energy_elbo_force_mae':
        energy_fn = partial(
            neg_var_elbo, 
            elbo=VariationalELBO(likelihood, model.gp, num_data=num_data, beta=beta)
        )
        force_fn = torch_mae
        loss_fn = partial(
            comb_force_loss, 
            energy_fn=energy_fn, 
            force_fn=force_fn,
            rho_force=rho_force
        )
    elif type == 'force' and loss_name == 'energy_mae_force_rmse':
        loss_fn = partial(
            force_loss, 
            rho_force=rho_force,
            energy_loss_type="mae",
            force_loss_type="rmse"
        )
    
    if type == 'energy' and model_class == 'UQ':
        wrapped_model = EnergyModel(model, loss_fn, likelihood)
    elif type == 'force' and model_class == 'UQ':
        wrapped_model = ForceModel(model, loss_fn, likelihood)
    elif type == 'energy' and model_class == 'base':
        wrapped_model = BaseEnergyModel(model, loss_fn)
    elif type == 'force' and model_class == 'base':
        wrapped_model = BaseForceModel(model, loss_fn)
    else:
        return model
    return wrapped_model



    
    
