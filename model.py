import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim
import numpy as np


class PlugInEstimator(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 covariate_samples: np.ndarray,
                 lr: float = 0.1,
                 monte_carlo_size: float = 32) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.monte_carlo_size = monte_carlo_size
        self.covariate_samples = torch.from_numpy(
            covariate_samples) if covariate_samples is not None else None
        self.val_idx_to_name = ["train", "val", "val_poi"]
        self.test_idx_to_name = ["test", "test_poi"]

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
        prefix = self.val_idx_to_name[dataloader_idx]
        return self._eval_loss_and_metric(
            batch,
            f"loss/{prefix}",
            f"metric/{prefix}",
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = self.test_idx_to_name[dataloader_idx]
        return self._eval_loss_and_metric(
            batch,
            f"loss/{prefix}",
            f"metric/{prefix}",
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
                                (self.monte_carlo_size,))
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

    def _eval_loss_and_metric(self, batch, loss_text_label, metric_text_label):
        y_hat_causal = self._estimate_causal_effect(batch)
        y_hat = self._predict_outcome(batch)
        y = batch[1]
        loss = self._loss_fn(y_hat, y)
        metric = self._metric_fn(y_hat_causal, y)
        metrics = {loss_text_label: loss, metric_text_label: metric}
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
    if model_type == "graph_smooth_linear_regression":
        return nn.Sequential(
            GraphSmoothLayer(graph_prior, **kwargs),
            nn.Linear(input_dim, 1),
        )
    elif model_type == "linear_regression":
        return nn.Linear(input_dim, 1, **kwargs)
    elif model_type == "graph_smooth_mlp":
        hidden_dim = kwargs.pop("hidden_dim", 32)
        return nn.Sequential(
            GraphSmoothLayer(graph_prior, **kwargs),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    elif model_type == "mlp":
        hidden_dim = kwargs.get("hidden_dim", 32)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    elif model_type == "graph_regularized_linear_regression":
        return GraphRegularizedLinearModel(graph_prior, input_dim, **kwargs)


class GraphSmoothLayer(nn.Module):

    def __init__(self, adj_matrix, n_steps=1, lazy_rate=0.):
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.from_numpy(adj_matrix)
        adj_matrix.fill_diagonal_(1)
        self.adj_matrix = adj_matrix
        self.n_steps = n_steps
        self.lazy_rate = lazy_rate
        self.diffusion_matrix = self._get_diffusion_matrix()

    def _get_diffusion_matrix(self):
        if self.n_steps == 0:
            return torch.eye(self.adj_matrix.shape[0])

        graph_laplacian = _compute_normalized_laplacian(self.adj_matrix)
        diffusion_matrix = self.lazy_rate * torch.eye(
            self.adj_matrix.shape[0]) + (1 - self.lazy_rate) * graph_laplacian
        return torch.matrix_power(diffusion_matrix, self.n_steps)

    def forward(self, h):
        x = h[..., :self.adj_matrix.shape[0]]
        z = h[..., self.adj_matrix.shape[0]:]
        x = torch.mm(self.diffusion_matrix, x)
        return torch.cat((x, z), dim=-1)


def _compute_normalized_laplacian(adj_matrix):
    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
    degree_sqrt_inv = torch.diag(torch.pow(torch.sum(adj_matrix, dim=1), -0.5))
    laplacian = degree_matrix - adj_matrix
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
        self.laplacian = torch.diag(torch.sum(graph_prior, dim=1)) - graph_prior
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
