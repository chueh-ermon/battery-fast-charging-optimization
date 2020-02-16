
# Closed-loop optimization of fast-charging protocols for batteries with machine learning

This repository contains much of the data, code, and visualizations associated with the paper:

> Closed-loop optimization of fast-charging protocols for batteries with machine learning  
> Peter Attia\*, Aditya Grover\*, Norman Jin, Kristen Severson, Todor Markov, Yang-Hung Liao, Michael Chen, Bryan Cheong, Nicholas Perkins, Zi Yang, Patrick Herring, Muratahan Aykol, Stephen Harris, Richard Braatz, Stefano Ermon, William Chueh  
> *Nature*, 2020  

\* equal contribution

The codebase is implemented in Python 3.7.


## Examples

Below we provide some example commands to run the key source files.

1. Defining a policy space.

```
python policies.py data all
```

The above command will dump a list of charging protocols (one per line) in `data/policies_all.csv` as well as a surface plot in `data/surface_all.png` for visualization of the protocol space.

2. Running a stochastic simulator (for e.g., hyperparameter optimization).

```
python sim_with_seed.py --C1=3.6 --C2=6 --C3=5.6 --seed=1
```

The above command will simulate a lifetime for the charging protocol CC1=3.6, CC2=6, CC3=5.6 with a seed of 1. Change the seed to observe more sampled lifetimes.


3. Running closed-loop optimization (CLO).

```
python closed_loop_oed.py --round=0 --data_dir='data' --next_batch_dir='batch' --arm_bounds_dir='pred' --early_pred_dir='batch'
```

This script will run CLO for first round and dump the suggested charging protocols for experimentation in `data/batch`. The log for this round will be available in `data/log.csv`. The lifetime estimates at the end of the current round are saved in `data/`. Before running CLO for next round, please ensure that the early predictions are saved in `data/pred/`.

See the arguments description in `closed_loop_oed.py` for further options.


## Paper Figures 

All the data and code for generating the figures in the paper are available in the `figures/` folder.


# Note

The primary contributors to this repository are [Peter Attia](https://github.com/petermattia) and [Aditya Grover](https://github.com/aditya-grover). The legacy repository with all the detailed experimental scripts (some of which uses internal infrastructure) is located [here](https://github.com/petermattia/battery-parameter-spaces). Associated repositories include [Arbin test file automation](https://github.com/chueh-ermon/automate-Arbin-schedule-file-creation) and [data processing and early prediction modeling](https://github.com/chueh-ermon/BMS-autoanalysis).
