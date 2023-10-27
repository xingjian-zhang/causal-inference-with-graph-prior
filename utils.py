import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_data(log_dir='tb_logs'):
    results = {}

    for root, dirs, _ in os.walk(log_dir):
        versions = sorted((d for d in dirs if d.startswith('version_')),
                          key=lambda x: int(x.split('_')[1]))

        for version in versions:  # Loop over all versions
            parts = root.split(os.sep)
            model, experiment = parts[-2:]

            acc = EventAccumulator(os.path.join(root, version))
            acc.Reload()
            scalars = {tag: acc.Scalars(tag) for tag in acc.Tags()['scalars']}
            results.setdefault(
                (model, version),
                {})[experiment] = scalars  # Include version in the key

    data = [
        {
            'model': model,
            'version': version,  # Add version info
            'experiment': experiment,
            'tag': tag,
            'step': value.step,
            'wall_time': value.wall_time,
            'value': value.value
        } for (model, version), experiments in results.items()
        for experiment, tags in experiments.items()
        for tag, values in tags.items() for value in values
    ]

    return pd.DataFrame(data)


def collect_results():
    df = load_data()
    df_filtered = df[df['tag'].str.contains('test')]
    max_steps = df_filtered.groupby(['model', 'experiment',
                                     'tag'])['step'].max()
    df_filtered = df[df['tag'].str.contains('test')]

    idx = df_filtered.groupby(['model', 'experiment', 'tag'])['step'].idxmax()
    df_filtered = df_filtered.loc[idx]

    df_pivot = df_filtered.pivot_table(index=['model', 'experiment'],
                                       columns='tag',
                                       values='value')
    model_map = {
        "graph_regularized_linear_regression": "GRLR",
        "linear_regression": "LR",
        "mlp": "MLP"
    }
    experiment_map = {"no_control": "NoCtrl", "budget_runtime": "CtrlCov"}
    df_pivot.index = df_pivot.index.map(lambda x:
                                        (model_map[x[0]], experiment_map[x[1]]))

    df_pivot_sorted = df_pivot.sort_values(by='metric/test_poi', ascending=True)
    selected_columns = df_pivot_sorted[['metric/test', 'metric/test_poi']]

    plt.figure(dpi=300)
    sns.set(rc={'figure.figsize': (10, 5)}, font_scale=0.8)
    ax = selected_columns.plot(kind='barh',
                               title='Test Accuracy (MSE)',
                               colormap='cividis')
    ax.legend(fontsize=10)
    for p in ax.patches:
        ax.text(p.get_width(),
                p.get_y() + p.get_height() / 2,
                f'{p.get_width():.3f}',
                va='center',
                fontsize=10)

    ax.set_xlabel('MSE', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/test_accuracy.png', dpi=300)

    df_pivot_sorted.to_csv('results/test_accuracy.tsv',
                           sep='\t',
                           index=True,
                           float_format='%.3f')
    print(df_pivot_sorted.to_markdown(floatfmt='.3f'))


class StreamToLogger():
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass