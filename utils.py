import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def load_data(log_dir="tb_logs"):
    results = {}

    for root, dirs, _ in os.walk(log_dir):
        versions = sorted((d for d in dirs if d.startswith("version_")),
                          key=lambda x: int(x.split("_")[1]))

        for version in versions:  # Loop over all versions
            parts = root.split(os.sep)
            model, experiment = parts[-2:]

            acc = EventAccumulator(os.path.join(root, version))
            acc.Reload()
            scalars = {tag: acc.Scalars(tag) for tag in acc.Tags()["scalars"]}
            results.setdefault(
                (model, version),
                {})[experiment] = scalars  # Include version in the key

    data = [
        {
            "model": model,
            "version": version,  # Add version info
            "experiment": experiment,
            "tag": tag,
            "step": value.step,
            "wall_time": value.wall_time,
            "value": value.value
        } for (model, version), experiments in results.items()
        for experiment, tags in experiments.items()
        for tag, values in tags.items() for value in values
    ]

    return pd.DataFrame(data)


def collect_results():
    df = load_data()
    df_filtered = df[df["tag"].str.contains("test")]

    idx = df_filtered.groupby(["model", "experiment", "tag", "version"])["step"].idxmax()
    df_filtered = df_filtered.loc[idx]
    df_filtered_gb = df_filtered.groupby(["model", "experiment", "tag"])

    # Calculate mean and standard deviation
    df_filtered_mean = df_filtered_gb.mean()
    df_filtered_std = df_filtered_gb.std() / np.sqrt(df_filtered_gb.count().values.min())

    df_pivot_mean = df_filtered_mean.pivot_table(index=["model", "experiment"],
                                                columns="tag",
                                                values="value")
    df_pivot_std = df_filtered_std.pivot_table(index=["model", "experiment"],
                                            columns="tag",
                                            values="value")

    model_map = {
        "graph_regularized_linear_regression": "LR-GR",
        "linear_regression": "LR",
        "simple_graph_convolution": "SGC"
    }
    experiment_map = {"no_control": "w/o control", "budget_runtime": "w/control"}
    df_pivot_mean.index = df_pivot_mean.index.map(lambda x:
                                        f"{model_map[x[0]]} {experiment_map[x[1]]}")
    df_pivot_std.index = df_pivot_std.index.map(lambda x:
                                        f"{model_map[x[0]]} {experiment_map[x[1]]}")

    # df_pivot_sorted = df_pivot_mean.sort_values(by="metric/test_poi", ascending=False)
    selected_columns_mean = df_pivot_mean[["metric/test", "metric/test_poi"]]
    selected_columns_std = df_pivot_std[["metric/test", "metric/test_poi"]]

    plt.figure(dpi=300)
    plt.tight_layout()
    sns.set(rc={"figure.figsize": (10, 5)}, font_scale=1)
    ax = selected_columns_mean.plot(kind="barh",
                                    xerr=selected_columns_std,
                                    title="Test Accuracy (MSE)",
                                    # colormap="cividis",
                                    capsize=4)  # capsize sets the width of the error bar caps

    ax.legend(["Test population", "Population of interest (distribution shift)"], loc="upper left", fontsize=10)

    for p in ax.patches:
        ax.text(p.get_width() - 0.15,
                p.get_y() + p.get_height() / 2,
                f"{p.get_width():.3f}",
                va="center",
                fontsize=10)

    ax.set_xlabel("MSE", fontsize=12)

    plt.savefig("results/test_accuracy.png", dpi=300, bbox_inches="tight")

    df_pivot_mean.to_csv("results/test_accuracy.tsv",
                        sep="\t",
                        index=True,
                        float_format="%.3f")
    print(df_pivot_mean.to_markdown(floatfmt=".3f"))


class StreamToLogger():
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass