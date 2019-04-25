import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("23_input_data.csv", sep = ";")

cols = df.columns

c = df[cols[-1]].tolist()

if __name__ == '__main__':
    # Choose window width and threshold
    window = 25
    thres = 250000

    # Iterate and collect state changes with regard to previous state
    changes = []
    rolling = [None] * window
    old_state = None
    for i in range(window, len(c) - 1):
        slc = c[i - window:i + 1]
        mean = sum(slc) / float(len(slc))
        state = 'good' if mean > thres else 'bad'

        rolling.append(mean)
        if not old_state or old_state != state:
            print('Changed to {:>4s} at position {:>3d} ({:5.3f})'.format(state, i, mean))
            changes.append((i, state))
            old_state = state

    # Plot results and state changes
    plt.figure(frameon=False, figsize=(10, 8))
    currents, = plt.plot(c, ls='--', label='Current')
    rollwndw, = plt.plot(rolling, lw=2, label='Rolling Mean')
    plt.axhline(thres, xmin=.0, xmax=1.0, c='grey', ls='-')
    plt.text(40, thres, 'Threshold: {:.1f}'.format(thres), horizontalalignment='right')
    for c, s in changes:
        plt.axvline(c, ymin=.0, ymax=.7, c='red', ls='-')
        plt.text(c, 41.5, s, color='red', rotation=90, verticalalignment='bottom')
    plt.legend(handles=[currents, rollwndw], fontsize=11)
    plt.grid(True)
    plt.savefig('plot.png', dpi=72, bbox_inches='tight')