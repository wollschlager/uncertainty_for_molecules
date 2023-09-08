from cmath import inf
import gpytorch
import torch
import seml 
import os
import re
import logging
import wandb
from torch.autograd import grad
from sacred import Experiment
from datetime import datetime
from os.path import join
from tqdm import tqdm
from ue4mol.models.lnk import MultiGPModel, Fixed_IndPoint_GPModel
from ue4mol.models.gp_base import Standard_GPModel
from ue4mol.utils import mae, rmse
from ue4mol.models.utils import load_global_backbone, load_localized_backbone
from ue4mol.models.utils import init_kernel, calc_inducing_points
from ue4mol.datasets.data_provider import get_dataloader
from ue4mol.train.optimizer import scheduler_map, EarlyStopping

ex = Experiment()
seml.setup_logger(ex)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
                db_collection, overwrite=overwrite))


    
@ex.automain
def run(_config, encoder_name, uq_name, uq_params, dataset, data_seed, num_epochs, kernel_name,
        batch_size, target, logdir, learning_rate, num_datapoints=None, freeze_encoder=True, learning_rate_after_unfreeze=None, kernel_params={},
        ood_datasets=None, normalize=True, encoder_params={}, scheduler=None, scheduler_params=None, debug=False, pretrained=None, use_mult_encoder=False,
        unfreeze_after=None, beta=1.0, batch_normalization=False, prediction_type='energy', rho_force=None, val_batch_size=None, 
        num_gps=None, dataset_params={}, fixed_ind_point=False, opt_func=None, inducing_point_initialization="kmeans", uq_lambda=1, 
        early_stopping_params={}, create_eps_env=False, eps_env_params={}):
    torch.backends.cuda.matmul.allow_tf32 = False
    if not 'patience' in early_stopping_params: 
        if 'patience' in scheduler_params:
            early_stopping_params['patience'] = 2 * scheduler_params['patience']
        else:
            early_stopping_params['patience'] = 100
    local_vars = locals()
    config = {}
    for k in local_vars.keys():
        if not k.startswith('_'):
            config[k] = local_vars[k]
    n_inducing_points = uq_params['n_inducing_points']
    print(n_inducing_points)
    logging.getLogger().setLevel(logging.DEBUG)
    run_id = _config['overwrite']
    db_collection = _config['db_collection']
    directory = join(logdir, dataset, db_collection, 
                     datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(run_id))
    logging.info(f"Directory: {directory}")
    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)

    wandb.init(
        dir=os.path.abspath(directory), 
        name=join(str(db_collection), str(run_id)),
        project='uncertainty-molecules',
        config=config
    )
    wandb.define_metric("epoch")
    wandb.define_metric('val_energy_mae', summary='min')
    wandb.define_metric('val_force_mae', summary='min')
    wandb.define_metric('val_loss', summary='min')
    wandb.define_metric('val_energy_mae_epoch', summary='min', step_metric="epoch")
    wandb.define_metric('val_force_mae_epoch', summary='min', step_metric="epoch")
    wandb.define_metric('val_loss_epoch', summary='min', step_metric="epoch")
    wandb.define_metric('train_energy_mae')
    wandb.define_metric('train_force_mae')
    wandb.define_metric('train_loss')
    
    train_loader, val_loader, _, _ = get_dataloader(
        dataset_name=dataset, 
        target=target,
        seed=data_seed,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        debug=debug,
        normalizing=normalize,
        num_datapoints=num_datapoints,
        **dataset_params
    )
    
    if pretrained is not None:
        if len(re.findall(".ckpt", pretrained)) == 0 and len(re.findall(".pth", pretrained)) == 0:
            pretrained = join(pretrained, dataset, encoder_name + '_' + str(target) + '_0.cpkt')
        
    if use_mult_encoder or uq_name == "mult-gp":
        model = load_localized_backbone(
            model_name=encoder_name,
            model_params=encoder_params,
            pretrained=pretrained
        )
    else:
        model = load_global_backbone(
            model_name=encoder_name,
            model_params=encoder_params,
            pretrained=pretrained
        )
    
    model = model.to(device)
    model.to_encoder()
    if freeze_encoder:
        model.freeze()   
        model.eval()
    else: 
        model.train()
    
    if uq_name == "mult-gp" and inducing_point_initialization != "mult-gp":
        print("CAREFUL WITH THE INDUCING POINT INITIALIZATION")
    inducing_points, energies, initial_lengthscale = calc_inducing_points(
        model=model, 
        train_loader=train_loader, 
        type=inducing_point_initialization, 
        n_inducing_points=uq_params['n_inducing_points']
    )

    kernel = init_kernel(kernel_name, kernel_params)
    kernel.lengthscale = initial_lengthscale.to(device) * torch.ones_like(kernel.lengthscale, device=device)
    if kernel_name == 'spectral_mixture':
        kernel.initialize_from_data_empspect(inducing_points, energies)
        covar_module = kernel
    else:
        covar_module = gpytorch.kernels.ScaleKernel(kernel)
    
    if uq_name == "gp":
        gp = Standard_GPModel(inducing_points, covar_module).to(device)
    elif uq_name in ['mult-gp']:
        if fixed_ind_point:
            single_gp = Fixed_IndPoint_GPModel(inducing_points, covar_module).to(device)
        else:
            single_gp = Standard_GPModel(inducing_points, covar_module).to(device)
        gp = MultiGPModel(single_gp, **uq_params)
        print('using multiple GPs')
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': gp.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    if scheduler:
        scheduler = scheduler_map[scheduler](optimizer, **scheduler_params)

    early_stopping = EarlyStopping(**early_stopping_params)
    best_loss, best_emae, best_fmae = inf, inf, inf


    gp.train()
    likelihood.train()
    # init mll 
    if opt_func is None:
        mll_func = gpytorch.mlls.VariationalELBO
    elif opt_func == 'variational_elbo':
        mll_func = gpytorch.mlls.VariationalELBO
    elif opt_func == 'predictive_log_likelihood':
        mll_func = gpytorch.mlls.PredictiveLogLikelihood
    else:
        raise NotImplementedError
    if uq_name in ['mult-gp', 'featurewise_gp']:
        mll = mll_func(likelihood, gp.gp, num_data=len(train_loader.dataset))
    else:
        mll = mll_func(likelihood, gp, num_data=len(train_loader.dataset))
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    losses = []
    val_losses = []
    for i in epochs_iter:
        wandb.log({'epoch': i})
        # train
        gp.train()
        likelihood.train()
        if unfreeze_after:
            if i >= unfreeze_after:
                print('unfreezing')
                model.unfreeze()
                freeze_encoder = False
                if learning_rate_after_unfreeze:
                    print('changing the learning rate')
                    learning_rate = learning_rate_after_unfreeze
                    for g in optimizer.param_groups:
                        g['lr'] = learning_rate
                        print(f"new learning rate for is {g['lr']}")
        if not freeze_encoder:
            model.train()
        minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False, disable=True)
        for batch in minibatch_iter:
            optimizer.zero_grad()

            batch.pos.requires_grad = True
            energy = gp(model(batch.cuda()), batch=batch.batch)
            forces = - grad(
                outputs=energy.mean.sum(),
                inputs=batch.pos,
                create_graph=True,
                retain_graph=True
            )[0]
            loss = (-mll(energy, batch.energy.cuda(), beta=beta) * (1 - rho_force) + rho_force * rmse(forces, batch.force)).mean()
            
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            wandb.log({
                'train_loss': loss.item(),
                'train_energy_mae': mae(energy.mean, batch.energy).mean().item(),
                'train_force_mae': mae(forces, batch.force).mean().item(),
            })

        # val
        gp.eval()
        likelihood.eval()
        model.eval()
        val_loss_epoch, val_energy_mae_epoch, val_force_mae_epoch = [], [], []
        minibatch_iter = tqdm(val_loader, desc="Val_Minibatch", leave=False, disable=True)
        for batch in minibatch_iter:
            optimizer.zero_grad()
            batch.pos.requires_grad = True
            energy = gp(model(batch.cuda()), batch=batch.batch)
            forces = - grad(
                outputs=energy.mean.sum(),
                inputs=batch.pos,
                create_graph=True,
                retain_graph=True
            )[0]
            loss = (-mll(energy, batch.energy.cuda(), beta=beta) * (1 - rho_force) + rho_force * rmse(forces, batch.force)).mean()
            minibatch_iter.set_postfix(loss=loss.item())
            val_losses.append(loss.item())
            val_loss_epoch.append(loss.item())
            val_energy_mae_epoch.append(mae(energy.mean, batch.energy).detach().cpu())
            val_force_mae_epoch.append(mae(forces, batch.force).mean().detach().cpu())
            wandb.log({
                'val_loss': loss.item(),
                'val_energy_mae': mae(energy.mean, batch.energy).mean().item(),
                'val_force_mae': mae(forces, batch.force).mean().item(),
            })
        avg_loss = torch.mean(torch.tensor(val_loss_epoch)).item()
        avg_emae = torch.mean(torch.tensor(val_energy_mae_epoch)).item()
        avg_fmae = torch.mean(torch.tensor(val_force_mae_epoch)).item()
        wandb.log({
            'val_energy_mae_epoch': avg_emae,
            'val_force_mae_epoch': avg_fmae,
            'val_loss_epoch': avg_loss
        })
        if scheduler:
            scheduler.step(avg_loss)
        
        early_stopping(avg_loss)
        
        wandb.log({
            'learning_rate': scheduler.optimizer.param_groups[0]['lr']
        })
        # save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(gp.state_dict(), join(directory, f'best_loss_gp.pth'))
            torch.save(likelihood.state_dict(), join(directory, f'best_loss_likelihood.pth'))
            torch.save(model.state_dict(), join(directory, f'best_loss_encoder.pth'))
        if avg_emae < best_emae:
            best_emae = avg_emae
            torch.save(gp.state_dict(), join(directory, f'best_e_mae_gp.pth'))
            torch.save(likelihood.state_dict(), join(directory, f'best_e_mae_likelihood.pth'))
            torch.save(model.state_dict(), join(directory, f'best_e_mae_encoder.pth'))     
        if avg_fmae < best_fmae:
            best_fmae = avg_fmae
            torch.save(gp.state_dict(), join(directory, f'best_f_mae_gp.pth'))
            torch.save(likelihood.state_dict(), join(directory, f'best_f_mae_likelihood.pth'))
            torch.save(model.state_dict(), join(directory, f'best_f_mae_encoder.pth'))     

        if early_stopping.early_stop:
            print("Early Stopping")
            break
    # save model -> update to best model maybe
    torch.save(gp.state_dict(), join(directory, 'gp.pth'))
    torch.save(likelihood.state_dict(), join(directory, 'likelihood.pth'))
    torch.save(model.state_dict(), join(directory, 'encoder.pth'))

    metrics = {
        'validation': {
            'best_energy_mae': best_emae,
            'best_forces_mae': best_fmae,
            'best_loss': best_loss
        }
    }
    return metrics
    


    


