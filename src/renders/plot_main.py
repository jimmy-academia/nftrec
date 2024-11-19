import random
import logging
from utils import loadj
from pathlib import Path
from .plot_functions import output_dir, rainbow_bar_plot, make_legend
from arguments import nft_project_names, plot_colors, Breeding_Types_Short, Breeding_Types, Baseline_Methods

def print_main():
    out_dir = output_dir/'main_exp'
    out_dir.mkdir(parents=True, exist_ok=True)

    taglist = ['revenue', 'buyer_utility', 'runtime']
    yaxis_list = ['Revenue', 'Avg. Utility', 'Runtime (s)']
    for tag, ylabel in zip(taglist, yaxis_list):
        for nft_project_name in nft_project_names:
            filename = f'{tag}_{nft_project_name}.jpg'
            filepath = out_dir/filename

            if filepath.exists():
                logging.info(f'|> {filepath} exists <|')
                continue
            else:
                results = read_info(tag, nft_project_name)
                y_axis_max = max(max(breeding_values) for breeding_values in results)
                y_axis_max += 0.1 * y_axis_max
                y_axis_min = min(min(breeding_values) for breeding_values in results) if tag == 'runtime' else 0
                y_axis_min -= 0.1 * y_axis_min

                infos = {
                    'log': tag == 'runtime',
                    'ylabel': ylabel,
                    'y_axis_max': y_axis_max,
                    'y_axis_min': y_axis_min,
                    'colors': plot_colors,
                    'xticks': Breeding_Types_Short,
                }

                rainbow_bar_plot(results, infos, filepath)
    legend_file = out_dir/'legend.jpg'
    legend_file.exists() and make_legend(Baseline_Methods, legend_file, 'bar', plot_colors)

def read_info(tag, nft_project_name):
    results = []
    for breeding_type in Breeding_Types:
        breeding_values = []
        for method in Baseline_Methods:
            rjson = Path(f'ckpt/main_exp/{nft_project_name}_{method}_{breeding_type}.json')
            if rjson.exists():
                data = loadj(rjson)
                if tag == 'revenue':
                    breeding_values.append(data.get('seller_revenue'))
                elif tag == 'buyer_utility':
                    breeding_values.append(data.get('avg_buyer_utility'))
                elif tag == 'runtime':
                    breeding_values.append(data.get('runtime'))
            else:
                breeding_values.append(-2 +random.random())
        results.append(breeding_values)

    return results