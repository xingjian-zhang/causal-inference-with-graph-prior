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
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    poi: np.ndarray
    graph_prior: np.ndarray


def synthetic_data_generation(
        orignal_filename: str = "data/clean_data.json",
        output_filename: str = "data/synthetic_data.json",
        num_cast=900,
        threshold_large_half=0.05,
        threshold_small_half=0.01,
        target_movie_set={5, 6, 10},
        noise_sigma=1,
        counterfactual=False,
        b=None,
        g=None,
        C=None):
    assert (counterfactual is not False and b is not None and g is not None
            and C is not None) or (counterfactual is False)
    np.random.seed(42)
    df = pd.read_json(orignal_filename)
    G = df['genres'].apply(lambda x: int(bool(set(x) & target_movie_set)))
    G_one_hot = pd.get_dummies(G, prefix='G').values

    if b is None:
        b = np.random.rand(num_cast, 1)
    if g is None:
        g = np.random.rand(2, 1)
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

    Y = np.dot(A, b) + np.dot(G_one_hot, g) + C
    Y = Y + np.random.normal(loc=0, scale=noise_sigma, size=(len(Y), 1))

    new_data = pd.DataFrame({
        'A': A,
        'G': G_one_hot.tolist(),
        'Y': Y.flatten()
    })
    syn_data = df.drop(columns=['revenue', 'cast', 'genres'])
    syn_data['cast'] = new_data['A'].apply(
        lambda x: [i for i, val in enumerate(x) if val == 1])
    syn_data['genres'] = new_data['G'].apply(
        lambda x: [i for i, val in enumerate(x) if val == 1])
    syn_data['revenue'] = new_data['Y']

    # save data
    syn_data.to_json(output_filename)
    np.savetxt("data/b_coefficient.dat", b)
    np.savetxt("data/g_coefficient.dat", g)
    np.savetxt("data/c_coefficient.dat", C)

    return syn_data, (b, g, C)


def get_synthetic_dataset_with_gprior(
    filename: str = "data/synthetic_data.json",
    poi: str = "genres",
    covariates: list = None,
    strategy: str = "gaussian_kernel",
    counterfactual=False,
    b=None,
    g=None,
    C=None,
    **kwargs,
) -> ObservationalDataWithGPrior:

    df, (b, g, C) = synthetic_data_generation(output_filename=filename,
                                              counterfactual=counterfactual,
                                              b=b,
                                              g=g,
                                              C=C)
    n_cast = df["cast"].apply(max).max() + 1

    def index_to_one_hot(index: list):
        vec = np.zeros(n_cast, dtype=int)
        vec[index] = 1
        return vec

    df["X"] = df["cast"].apply(index_to_one_hot)
    x = np.stack(df["X"].values)
    y = df["revenue"].values
    z = _get_covariates(df, covariates)
    z = _standardize(z)

    if poi == "genres":
        poi = df['genres'].apply(lambda x: x[0] == 0)
        poi = poi.values
    else:
        raise NotImplementedError(f"POI {poi} not implemented.")

    kwargs['b'] = b
    graph_prior = get_graph_prior(df, strategy, **kwargs)
    x, y, z = _to_dtype(x, y, z, dtype=np.float32)

    return ObservationalDataWithGPrior(x, y, z, poi, graph_prior)


def get_obs_dataset_with_gprior(
    filename: str = "data/clean_data.json",
    poi: str = "genres",
    covariates: list = None,
    strategy: str = "genre_similarity",
    **kwargs,
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

    graph_prior = get_graph_prior(df, strategy, **kwargs)
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
    threshold: float = 0.8,
    binary: bool = True,
    **kwargs,
):
    b = kwargs.get('b')
    n_cast = df["cast"].apply(max).max() + 1
    assert len(b) == n_cast
    actor_similarity_matrix = np.zeros((n_cast, n_cast), dtype=float)
    bi = np.tile(b, (n_cast, 1))
    bj = bi.T
    actor_similarity_matrix = np.exp(-((bi - bj)**2))  # sigma = 1

    # when memory not enough:
    # Define the Gaussian kernel function
    # def gaussian_kernel(u, v, sigma=1):
    #     return np.exp(-((u - v) ** 2) / sigma ** 2)
    # for i in range(n_cast):
    #     for j in range(i,n_cast):
    #         actor_similarity_matrix[i, j] = gaussian_kernel(b[i], b[j]).item()
    #         actor_similarity_matrix[j, i] = actor_similarity_matrix[i,j]
    if binary:
        return np.where(actor_similarity_matrix > threshold, 1, 0)
    else:
        return np.where(actor_similarity_matrix > threshold,
                        actor_similarity_matrix, 0)


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
