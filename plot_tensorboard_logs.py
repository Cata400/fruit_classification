import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

log_name = 'log_tf_pca10_20230107-142526'

experiment_id = '41rd03V3R3Ofgv2aVbiATA'
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

interest_df = df[df['run'].str.contains(log_name)]
accuracy_df = interest_df[interest_df['tag'].str.contains('epoch_accuracy')]
loss_df = interest_df[interest_df['tag'].str.contains('epoch_loss')]

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
ax_acc = sns.lineplot(accuracy_df, x='step', y='value', hue=accuracy_df.run.apply(lambda x: x.split('/')[-1]))
ax_acc.set_title('Accuracy')
ax_acc.set(xlabel='Epoch', ylabel='Value')
handles, labels = ax_acc.get_legend_handles_labels()
ax_acc.legend(handles=handles, labels=labels)

plt.subplot(1, 2, 2)
ax_loss = sns.lineplot(loss_df, x='step', y='value', hue=loss_df.run.apply(lambda x: x.split('/')[-1]))
ax_loss.set_title('Loss')
ax_loss.set(xlabel='Epoch', ylabel='Value')
handles, labels = ax_loss.get_legend_handles_labels()
ax_loss.legend(handles=handles, labels=labels)

plt.show()
