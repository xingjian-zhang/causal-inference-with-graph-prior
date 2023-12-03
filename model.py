from typing import Dict, Union
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim
import numpy as np


class PlugInEstimator(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        covariate_samples: np.ndarray,
        lr: float = 0.1,
        monte_carlo_size: float = 32,
        true_params: Dict[str, Union[np.ndarray, float]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.monte_carlo_size = monte_carlo_size
        self.covariate_samples = torch.from_numpy(
            covariate_samples) if covariate_samples is not None else None
        self.val_idx_to_name = ["train", "val", "val_poi"]
        self.test_idx_to_name = ["test", "test_poi"]
        self.true_params = true_params

    def forward(self, x: torch.tensor, z: torch.tensor):
        if z is None:
            return self.model(x)
        h = torch.concat((x, z), dim=1)
        return self.model(h)

    def training_step(self, batch, batch_idx):
        x, y, z = self._unpack_batch(batch)
        y_hat = self.forward(x, z)
        loss = self._loss_fn(y_hat, y)
        if hasattr(self.model, "regularized_loss"):
            reg_loss = self.model.regularized_loss()
            loss += reg_loss
            self.log("loss/reg", reg_loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        suffix = self.val_idx_to_name[dataloader_idx]
        return self._eval_metrics(batch, suffix)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        suffix = self.test_idx_to_name[dataloader_idx]
        return self._eval_metrics(batch, suffix)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.lr,
                                weight_decay=1e-4)

    def _loss_fn(self, y_hat: torch.tensor, y: torch.tensor):
        return nn.functional.mse_loss(y_hat.squeeze(), y.squeeze())

    def _metric_fn(self, y_hat: torch.tensor, y: torch.tensor):
        return nn.functional.mse_loss(y_hat.squeeze(), y.squeeze())

    def _unpack_batch(self, batch):
        try:
            x, y, z = batch
        except ValueError:
            x, y = batch
            z = None
        return x, y, z

    def _sample_covariates(self):
        indices = torch.randint(0, self.covariate_samples.shape[0],
                                (self.monte_carlo_size, ))
        return self.covariate_samples[indices]

    def _predict_outcome(self, batch):
        x, _, z = self._unpack_batch(batch)
        return self.forward(x, z)

    def _estimate_causal_effect(self, batch):
        x, _, z = self._unpack_batch(batch)

        # Without covariates control, just return the prediction.
        if z is None:
            return self._predict_outcome(batch)

        # Control covariates using plug-in estimator.
        # E[Y_X] = E_Z[E[Y_X|X=x, Z]]
        # NOTE: we use the same covariates MC sample for all the data points.
        z_sample = self._sample_covariates().to(z.device)
        y_hats = []
        for z in z_sample:
            z_batch = z.repeat(x.shape[0], 1)
            y_hat = self.forward(x, z_batch)
            y_hats.append(y_hat.squeeze())
        y_hat_causal = torch.stack(y_hats).mean(
            dim=0)  # Average over MC samples.
        return y_hat_causal

    def _eval_params_mse(self):
        if self.true_params is None:
            raise ValueError("true_params is not provided.")

        true_params = torch.from_numpy(self.true_params["b"]).squeeze()
        params = _get_causal_params(self.model, len(true_params)).squeeze()
        true_params = true_params.to(device=params.device, dtype=params.dtype)

        mse = nn.functional.mse_loss(params, true_params)
        return mse

    def _eval_metrics(self, batch, suffix):
        y_hat_causal = self._estimate_causal_effect(batch)
        y_hat = self._predict_outcome(batch)
        y = batch[1]
        loss = self._loss_fn(y_hat, y)
        metric = self._metric_fn(y_hat_causal, y)
        metrics = {f"loss/{suffix}": loss, f"metric/{suffix}": metric}
        if suffix in ("val", "test") and self.true_params:
            metrics[f"params_mse/{suffix}"] = self._eval_params_mse()

        for k, v in metrics.items():
            self.log(k,
                     v,
                     on_step=False,
                     on_epoch=True,
                     logger=True,
                     add_dataloader_idx=False)
        return metrics


def model_factory(input_dim: int,
                  graph_prior: np.ndarray,
                  model_type: str = "linear_regression",
                  **kwargs):
    if model_type == "linear_regression":
        return nn.Linear(input_dim, 1, **kwargs)
    elif model_type == "graph_regularized_linear_regression":
        return GraphRegularizedLinearModel(graph_prior, input_dim, **kwargs)
    elif model_type == "simple_graph_convolution":
        return nn.Sequential(
            GraphSmoothLayer(graph_prior, **kwargs),
            nn.Linear(input_dim, 1),
        )


class GraphSmoothLayer(nn.Module):

    def __init__(self, adj_matrix, n_steps=1, lazy_rate=0.5):
        super().__init__()
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.from_numpy(adj_matrix)
        adj_matrix.fill_diagonal_(1)
        self.adj_matrix = adj_matrix
        self.n_steps = n_steps
        self.lazy_rate = lazy_rate
        self.diffusion_matrix = self._get_diffusion_matrix()

    def _get_diffusion_matrix(self):
        # Calculate row sums (degree of each vertex)
        deg = self.adj_matrix.sum(dim=1, keepdim=True)
        # Normalize adjacency matrix to get the transition matrix
        transition_matrix = self.adj_matrix / deg

        # Incorporate lazy rate
        identity = torch.eye(self.adj_matrix.size(0),
                             dtype=self.adj_matrix.dtype)
        diffusion_matrix = self.lazy_rate * identity + (
            1 - self.lazy_rate) * transition_matrix

        # If n_steps > 1, raise the matrix to the power of n_steps
        if self.n_steps > 1:
            diffusion_matrix = diffusion_matrix.matrix_power(self.n_steps)

        return diffusion_matrix

    def forward(self, h):
        if self.diffusion_matrix.dtype != h.dtype:
            self.diffusion_matrix = self.diffusion_matrix.to(h.dtype)
        if self.diffusion_matrix.device != h.device:
            self.diffusion_matrix = self.diffusion_matrix.to(h.device)
        x = h[..., :self.adj_matrix.shape[0]]
        z = h[..., self.adj_matrix.shape[0]:]
        x = torch.mm(x, self.diffusion_matrix)
        return torch.cat((x, z), dim=-1)


def _compute_normalized_laplacian(adj_matrix):
    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
    degree_sqrt_inv = torch.diag(torch.pow(torch.sum(adj_matrix, dim=1), -0.5))
    laplacian = degree_matrix - adj_matrix
    laplacian = laplacian.to(degree_sqrt_inv.dtype)
    normalized_laplacian = torch.mm(torch.mm(degree_sqrt_inv, laplacian),
                                    degree_sqrt_inv)
    return normalized_laplacian


class GraphRegularizedLinearModel(nn.Module):

    def __init__(self, graph_prior, input_dim, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        graph_prior = torch.from_numpy(graph_prior)
        self.graph_prior = graph_prior
        self.linear = nn.Linear(input_dim, 1)
        self.laplacian = torch.diag(torch.sum(graph_prior,
                                              dim=1)) - graph_prior
        self.laplacian = self.laplacian.to(self.linear.weight.dtype)

    def forward(self, h):
        return self.linear(h)

    def regularized_loss(self):
        if self.laplacian.device != self.linear.weight.device:
            self.laplacian = self.laplacian.to(self.linear.weight.device)
        params = self.linear.weight[:, :self.graph_prior.shape[0]]
        loss = params @ self.laplacian @ params.T
        loss = loss.squeeze()
        return self.alpha * loss


def _get_causal_params(model: nn.Module, num_params: int):
    linear = None
    if isinstance(model, GraphRegularizedLinearModel):
        linear = model.linear
    elif isinstance(model, nn.Linear):
        linear = model
    elif isinstance(model, nn.Sequential):  # SGC
        linear = model[-1]
    return linear.weight.squeeze()[:num_params]
