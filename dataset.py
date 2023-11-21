from typing import Dict, Any
import pandas as pd
import numpy as np
import torch
import dataclasses
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
from torch.utils.data import TensorDataset, DataLoader


@dataclasses.dataclass
class ObservationalDataWithGPrior:
    """Observational data.
    Attributes
        x: The multi-dimensional causes.
        y: The target estimand.
        z: The multi-dimensional covariates.
        poi: Population of interest.
        graph_prior: The graph prior adjacent matrix.
        parameters: The parameters used to generate the data, if synthetic.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    poi: np.ndarray
    graph_prior: np.ndarray
    parameters: dict = dataclasses.field(default_factory=dict)


def get_synthetic_dataset_with_gprior(
        original_filename: str = "data/clean_data.json",
        num_cast=900,
        threshold_large_half=0.05,
        threshold_small_half=0.01,
        target_movie_set={5, 6, 10},
        noise_sigma=1,
        counterfactual=False,
        b=None,
        g=None,
        C=None,
        random_seed=None,
        poi: str = "genres",
        covariates: bool = True,
        strategy: str = "gaussian_kernel",
        strategy_kwargs: Dict[str, Any] = None) -> ObservationalDataWithGPrior:
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    assert (counterfactual and b is not None and g is not None
            and C is not None) or (not counterfactual)
    df = pd.read_json(original_filename)
    target_movie_set = set(target_movie_set)
    G = df['genres'].apply(lambda x: int(bool(set(x) & target_movie_set)))
    G = G.values

    if b is None:
        b = np.random.rand(num_cast, 1)
    if g is None:
        g = np.random.rand(1)
    if C is None:
        C = np.random.rand(1)

    A = []
    for g_val in G:
        if g_val == (1 if not counterfactual else 0):
            first_half = np.random.choice(
                [0, 1],
                size=num_cast // 2,
                p=[1 - threshold_large_half, threshold_large_half])
            second_half = np.random.choice(
                [0, 1],
                size=num_cast // 2,
                p=[1 - threshold_small_half, threshold_small_half])
        else:
            first_half = np.random.choice(
                [0, 1],
                size=num_cast // 2,
                p=[1 - threshold_small_half, threshold_small_half])
            second_half = np.random.choice(
                [0, 1],
                size=num_cast // 2,
                p=[1 - threshold_large_half, threshold_large_half])
        A.append(list(first_half) + list(second_half))

    Y = np.dot(A, b) + g * G[:, None] + C
    Y = Y + np.random.normal(loc=0, scale=noise_sigma, size=(len(Y), 1))

    new_data = pd.DataFrame({
        'A': A,
        'G': G,
        'Y': Y.flatten()
    })
    syn_data = df.drop(columns=['revenue', 'cast', 'genres'])
    syn_data['cast'] = new_data['A'].apply(
        lambda x: [i for i, val in enumerate(x) if val == 1])
    syn_data['genres'] = new_data['G']
    syn_data['revenue'] = new_data['Y']

    df = syn_data
    n_cast = df["cast"].apply(max).max() + 1

    def index_to_one_hot(index: list):
        vec = np.zeros(n_cast, dtype=int)
        vec[index] = 1
        return vec

    df["X"] = df["cast"].apply(index_to_one_hot)
    x = np.stack(df["X"].values)
    y = df["revenue"].values
    z = G[:, None] if covariates else None

    if poi == "genres":
        poi = df['genres'].apply(lambda x: x == 0)
        poi = poi.values
    else:
        raise NotImplementedError(f"POI {poi} not implemented.")

    strategy_kwargs['b'] = b
    graph_prior = get_graph_prior(df, strategy, **strategy_kwargs)
    x, y, z = _to_dtype(x, y, z, dtype=np.float32)
    params = {"b": b, "g": g, "C": C}

    return ObservationalDataWithGPrior(x, y, z, poi, graph_prior, params)


def get_obs_dataset_with_gprior(
    filename: str = "data/clean_data.json",
    poi: str = "genres",
    covariates: list = None,
    strategy: str = "genre_similarity",
    strategy_kwargs: Dict[str, Any] = None,
) -> ObservationalDataWithGPrior:
    """Get the tabular dataset for TMDB5000.

    Args:
        filename: The path to the dataset.
        poi: The population of interest.
        covariates: The covariates to be used.
        strategy: The strategy to use. Default to genre_similarity.
        kwargs: The keyword arguments. Additional arguments for the strategy.

    Returns:
        The tabular dataset.
    """
    df = pd.read_json(filename)

    n_cast = df["cast"].apply(max).max() + 1

    def index_to_one_hot(index: list):
        vec = np.zeros(n_cast, dtype=int)
        vec[index] = 1
        return vec

    df["X"] = df["cast"].apply(index_to_one_hot)
    x = np.stack(df["X"].values)
    y = df["revenue"].values
    y = np.log(y)
    y = _standardize(y)
    z = _get_covariates(df, covariates)
    z = _standardize(z)

    # Population of interest
    if poi == "genres":
        poi = df["genres"].apply(lambda g: all(i not in g for i in (5, 10, 6)))
        poi = poi.values
    else:
        raise NotImplementedError(f"POI {poi} not implemented.")

    graph_prior = get_graph_prior(df, strategy, **strategy_kwargs)
    x, y, z = _to_dtype(x, y, z, dtype=np.float32)

    return ObservationalDataWithGPrior(x, y, z, poi, graph_prior)


def get_graph_prior(
    df: pd.DataFrame,
    strategy: str = "genre_similarity",
    **kwargs,
) -> np.ndarray:
    """Get the graph prior.

    Args:
        df: The dataframe.
        strategy: The strategy to use. Default to genre_similarity.
        kwargs: The keyword arguments. Additional arguments for the strategy.

    Returns:
        The graph prior adjacent matrix.
    """
    func_dict = {
        "genre_similarity": _get_genre_similarity_prior,
        "gaussian_kernel": _get_gaussian_kernel_prior
    }
    if strategy not in func_dict:
        raise NotImplementedError(f"Strategy {strategy} not implemented.")
    prior_graph = func_dict[strategy](df, **kwargs)
    return prior_graph


def split_dataset(obs_dataset: ObservationalDataWithGPrior,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1):
    """Split the dataset into train, validation and test sets.

    Args:
        obs_dataset: The observational dataset.
        train_ratio: The ratio of training set.
        val_ratio: The ratio of validation set.
        test_ratio: The ratio of test set.

    Returns:
        The split dataset.
    """
    graph_prior = obs_dataset.graph_prior
    train_arrays, val_arrays, test_arrays = _split_arrays(
        obs_dataset.x,
        obs_dataset.y,
        obs_dataset.z,
        obs_dataset.poi,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_dataset = ObservationalDataWithGPrior(*train_arrays, graph_prior)
    val_dataset = ObservationalDataWithGPrior(*val_arrays, graph_prior)
    test_dataset = ObservationalDataWithGPrior(*test_arrays, graph_prior)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(
    data: ObservationalDataWithGPrior,
    use_poi=False,
    shuffle=False,
    **kwargs,
):
    """Get the dataloader.

    Args:
        data: The observational dataset.
        kwargs: The keyword arguments. Additional arguments for the dataloader.

    Returns:
        The dataloader.
    """
    x = torch.from_numpy(data.x)
    y = torch.from_numpy(data.y)
    z = torch.from_numpy(data.z) if data.z is not None else None
    poi = torch.from_numpy(data.poi)
    arrays = []

    if z is not None:
        arrays = [x, y, z]
    else:
        arrays = [x, y]
    if use_poi:
        arrays = [array[poi] for array in arrays]

    return DataLoader(TensorDataset(*arrays), shuffle=shuffle, **kwargs)


def _split_arrays(
    *arrays,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    # Check that the ratios add up to 1.0
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must add up to 1.0"

    # Get the number of samples
    n_samples = arrays[0].shape[0]

    # Calculate the number of samples for each set
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    # Shuffle the indices
    indices = np.random.permutation(n_samples)

    # Split the indices into sets
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Split the arrays using the indices
    get_array = lambda indices: tuple(array[indices]
                                      if array is not None else None
                                      for array in arrays)
    train_arrays = get_array(train_indices)
    val_arrays = get_array(val_indices)
    test_arrays = get_array(test_indices)

    return train_arrays, val_arrays, test_arrays


def _get_gaussian_kernel_prior(
    df: pd.DataFrame,
    b: np.ndarray = None,
    threshold: float = 0.99,
    binary: bool = True,
    sigma: bool = None,
):
    n_cast = df["cast"].apply(max).max() + 1
    assert len(b) == n_cast
    actor_similarity_matrix = np.exp(-(b - b.T)**2 / sigma**2)

    if binary:
        graph_prior = np.where(actor_similarity_matrix > threshold, 1, 0)
    else:
        graph_prior = np.where(actor_similarity_matrix > threshold,
                               actor_similarity_matrix, 0)

    if binary:
        sparsity = np.count_nonzero(graph_prior) / graph_prior.size
        logging.info("Sparsity of graph_prior matrix %f", sparsity)

    return graph_prior


def _get_genre_similarity_prior(
    df: pd.DataFrame,
    threshold: float = 0.8,
    binary: bool = True,
):
    """Compute the genre similarity and clip the adjacent matrix."""
    n_cast = df["cast"].apply(max).max() + 1
    n_genre = max(max(genres) for genres in df['genres'] if len(genres)) + 1

    genre_counts = np.zeros((n_cast, n_genre), dtype=int)

    for _, row in df.iterrows():
        genres = row['genres']
        actor_vector = row['X']

        for actor_idx, is_actor_in_movie in enumerate(actor_vector):
            if is_actor_in_movie:
                for genre in genres:
                    genre_counts[actor_idx, genre] += 1

    # Standardize the genre_counts array
    genre_counts = _standardize(genre_counts)
    actor_similarity_matrix = cosine_similarity(genre_counts)

    # Construct graph prior
    if binary:
        return np.where(actor_similarity_matrix > threshold, 1, 0)
    else:
        return np.where(actor_similarity_matrix > threshold,
                        actor_similarity_matrix, 0)


def _get_covariates(df: pd.DataFrame, covariates: list):
    # NOTE: not implemented PCA deconfounder yet.
    if covariates is None or not covariates:
        return None

    covariates_in_df = list(set.intersection(set(covariates), set(df.columns)))
    if len(covariates_in_df) != len(covariates):
        logging.warning(
            f"Missing covariates: {set(covariates) - set(covariates_in_df)}.")
    return df[covariates_in_df].values


def _standardize(x):
    # Check if the input data x is 1D. If yes, reshape it to 2D.
    if x is None:
        return x

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    scaler = StandardScaler()
    x_standardized = scaler.fit_transform(x)

    # If the input was 1D, convert it back to 1D after standardization.
    if x_standardized.shape[1] == 1:
        x_standardized = x_standardized.flatten()

    return x_standardized


def _to_dtype(*arrays, dtype=np.float32):
    new_arrays = []
    for array in arrays:
        new_arrays.append(array.astype(dtype) if array is not None else None)
    return new_arrays


if __name__ == "__main__":
    obs_dataset = get_obs_dataset_with_gprior()
    print("Number of cast:", obs_dataset.x.shape[1])
    if obs_dataset.z is not None:
        print("Number of covariates:", obs_dataset.z.shape[1])
    # print("Graph prior", obs_dataset.graph_prior)
    print("Average degree of prior graph: {:.2f}".format(
        np.mean(np.sum(obs_dataset.graph_prior, axis=1))))
