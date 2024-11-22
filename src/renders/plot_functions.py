from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "font.size": 50,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
})

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
ckpt_dir = Path('ckpt')

def rainbow_bar_plot(project_revenues, infos, filepath, preview=False):
    """Creates a rainbow bar plot and saves it to a file.

    Args:
        project_revenues (list): List of project revenues.
        infos (dict): Information for the plot (e.g., colors, xticks, labels).
        filepath (str or Path): Path to save the plot.
    """
    plt.figure(figsize=(2.2, 1)) if preview else plt.figure(figsize=(13, 6), dpi=200)
    plt.ylabel(infos['ylabel'], fontweight='bold')
    plt.ylim(infos['y_axis_min'], infos['y_axis_max'])

    bar_width = 1
    set_width = 1.2*len(infos['colors']) + 0.6
    indexes = range(len(project_revenues))
    for index, rev_rainbow in zip(indexes, project_revenues):
        for k, (rev, color) in enumerate(zip(rev_rainbow, infos['colors'])):
            plt.bar(index*set_width+k*(bar_width+0.2), rev, bar_width, color=color)
        if index != len(project_revenues) - 1:
            plt.axvline(x=index*set_width+(k+1)*(bar_width+0.2)-0.2, color='black', linestyle=':', linewidth=3)
    
    if infos['xticks'] is not None:
        tick_positions = [index * set_width + (len(infos['colors']) - 1) / 2 * (bar_width + 0.2) for index in range(len(project_revenues))]
        plt.xticks(tick_positions, infos['xticks'], fontsize=30)
    else:
        plt.xticks([])

    if infos['log']:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()

def make_legend(legends, filepath, tag, colors, patterns=None, markers=None):

    """Creates a legend plot and saves it to a file.

    Args:
        legends (list): List of legend labels.
        filepath (str or Path): Path to save the legend.
        tag (str): Type of legend ('bar', 'line', 'tripple').
        colors (list): List of colors for the legend entries.
        patterns (list, optional): Patterns for 'tripple' legends. Defaults to None.
        markers (list, optional): Markers for 'line' legends. Defaults to None.
    """
    fig, ax = plt.subplots()
    if tag == 'bar':
        [ax.bar(0, 0, color=colors[i], label=legends[i]) for i in range(len(legends))]
    elif tag == 'line':
        [ax.plot(0, 0, color=colors[i], label=legends[i], marker=markers[i], markersize=30, linewidth=12)  for i in range(len(legends))]
    elif tag == 'tripple':
        num_breed = 4
        bars = [ax.bar(0,0, color=colors[i], label=legends[i]) for i in range(len(legends) - num_breed)]
        bars += [ax.bar(0,0, color='white', edgecolor='black', hatch=patterns[i], label=legends[i-num_breed]) for i in range(num_breed)]
    else:
        raise ValueError(f"Unsupported tag: {tag}")

    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)

    legend_fig_width = len(legends) * 0.5  # inches per entry, adjust as needed
    fig_legend = plt.figure(figsize=(legend_fig_width, 1), dpi=300)  
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=len(legends), frameon=False, 
        fontsize=50, handlelength=0.8, handletextpad=0.2, columnspacing=0.75, markerscale=1.2)
    fig_legend.savefig(filepath, bbox_inches='tight')
    plt.close(fig_legend)


