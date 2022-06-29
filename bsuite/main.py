import warnings

from bsuite.experiments import summary_analysis
from bsuite.logging import csv_load
from bsuite.logging import sqlite_load

import numpy as np
import pandas as pd
import plotnine as gg
#@title Import experiment-specific analysis

from bsuite.experiments.bandit import analysis as bandit_analysis
from bsuite.experiments.bandit_noise import analysis as bandit_noise_analysis
from bsuite.experiments.bandit_scale import analysis as bandit_scale_analysis
from bsuite.experiments.cartpole import analysis as cartpole_analysis
from bsuite.experiments.cartpole_noise import analysis as cartpole_noise_analysis
from bsuite.experiments.cartpole_scale import analysis as cartpole_scale_analysis
from bsuite.experiments.cartpole_swingup import analysis as cartpole_swingup_analysis
from bsuite.experiments.catch import analysis as catch_analysis
from bsuite.experiments.catch_noise import analysis as catch_noise_analysis
from bsuite.experiments.catch_scale import analysis as catch_scale_analysis
from bsuite.experiments.deep_sea import analysis as deep_sea_analysis
from bsuite.experiments.deep_sea_stochastic import analysis as deep_sea_stochastic_analysis
from bsuite.experiments.discounting_chain import analysis as discounting_chain_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
from bsuite.experiments.mnist import analysis as mnist_analysis
from bsuite.experiments.mnist_noise import analysis as mnist_noise_analysis
from bsuite.experiments.mnist_scale import analysis as mnist_scale_analysis
from bsuite.experiments.mountain_car import analysis as mountain_car_analysis
from bsuite.experiments.mountain_car_noise import analysis as mountain_car_noise_analysis
from bsuite.experiments.mountain_car_scale import analysis as mountain_car_scale_analysis
from bsuite.experiments.umbrella_distract import analysis as umbrella_distract_analysis
from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis

pd.options.mode.chained_assignment = None
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing_x=0.5, panel_spacing_y=0.5)
warnings.filterwarnings('ignore')

#@title loading results from local data:

experiments = {}  # Add results here
DF, SWEEP_VARS = sqlite_load.load_bsuite(experiments)
# Or
# DF, SWEEP_VARS = csv_load.load_bsuite(experiments)

#@title overall score as radar plot (double-click to show/hide code)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
__radar_fig__ = summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)

#@title plotting overall score as bar (double-click to show/hide code)
summary_analysis.bsuite_bar_plot(BSUITE_SCORE, SWEEP_VARS).draw();

#@title compare agent performance on each challenge (double-click to show/hide code)
summary_analysis.bsuite_bar_plot_compare(BSUITE_SCORE, SWEEP_VARS).draw();

#@title parsing data
bandit_df = DF[DF.bsuite_env == 'bandit'].copy()
summary_analysis.plot_single_experiment(BSUITE_SCORE, 'bandit', SWEEP_VARS).draw();

#@title plot average regret through learning (lower is better)
bandit_analysis.plot_learning(bandit_df, SWEEP_VARS).draw();

#@title plot performance by seed (higher is better)
bandit_analysis.plot_seeds(bandit_df, SWEEP_VARS).draw();

#@title parsing data
mountain_car_df = DF[DF.bsuite_env == 'mountain_car'].copy()
summary_analysis.plot_single_experiment(BSUITE_SCORE, 'mountain_car', SWEEP_VARS).draw();

#@title plot average regret through learning (lower is better)
mountain_car_analysis.plot_learning(mountain_car_df, SWEEP_VARS).draw();

#@title plot performance by seed (higher is better)
mountain_car_analysis.plot_seeds(mountain_car_df, SWEEP_VARS).draw();

#@title parsing data
bandit_noise_df = DF[DF.bsuite_env == 'bandit_noise'].copy()
summary_analysis.plot_single_experiment(BSUITE_SCORE, 'bandit_noise', SWEEP_VARS).draw();

#@title average regret over learning (lower is better)
bandit_noise_analysis.plot_average(bandit_noise_df, SWEEP_VARS).draw();

#@title average regret through learning (lower is better)
bandit_noise_analysis.plot_learning(bandit_noise_df, SWEEP_VARS).draw();

#@title plot performance by seed (higher is better)
bandit_noise_analysis.plot_seeds(bandit_noise_df, SWEEP_VARS).draw();

#@title parsing data
bandit_scale_df = DF[DF.bsuite_env == 'bandit_scale'].copy()
summary_analysis.plot_single_experiment(BSUITE_SCORE, 'bandit_scale', SWEEP_VARS).draw();

#@title average regret over learning (lower is better)
bandit_scale_analysis.plot_average(bandit_scale_df, SWEEP_VARS).draw();

#@title average regret through learning (lower is better)
bandit_scale_analysis.plot_learning(bandit_scale_df, SWEEP_VARS).draw();

#@title plot performance by seed (higher is better)
bandit_scale_analysis.plot_seeds(bandit_scale_df, SWEEP_VARS).draw();

