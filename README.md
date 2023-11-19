# Bodaway Gap Research

Research tools to analyze and design water infrastructure for the [Bodaway Gap](https://bodaway.navajochapters.org/) chapter of the Navajo nation. Part of a 3-year NSF-funded [contextual engineering](http://contextual.engineering.illinois.edu/) study under the University of Illinois [Applied Research Institute](https://appliedresearch.illinois.edu/). Read more [here](https://appliedresearch.illinois.edu/news/contextual-innovation-and-practice-group-receives-nfs-grant-to-address-navajo-nation-water-and-energy-needs).

> For more information, email [Quinn Ouyang](qouyang3@illinois.edu), [Abhiroop Chattopadhyay](ac33@illinois.edu), or [Ann Witmer](awitmer@illinois.edu)

## Projects

- `facility-clustering` - Early experimentation with different optimization models for reservoir assignment
- `dynamics-modeling` - Optimization model of dynamics for a specific reservoir. Eventually for modeling an entire system with demand, energy, etc. dynamics.
- `source-hydro` - Case study analysis of the ineffectiveness of Hydropanels in the region
- `travel-analysis` - Early experimentation with analyzing travel times between reservoirs and demand points

## Development

This project consists of several [Jupyter notebooks](https://jupyter.org/) for optimization modeling via [Pyomo](https://pyomo.readthedocs.io/en/stable/) and data processing with Pandas and NumPy.

### Local

1. Clone or fork the repository.

   ```
   git clone https://github.com/quinnouyang/Bodaway-Gap-Research.git
   ```

2. Ensure you have Python version `>=3.8` or later (we developed with `3.11.6` and `3.10`).

   ```
   python --version
   ```

3. Create and activate a new [virtual enviornment](https://docs.python.org/3/tutorial/venv.html).

   ```
   python -m venv <venv-name>
   source <venv-name>/bin/activate
   ```

4. Install dependencies
   ```
   pip install -r requirements.txt
   ```

### Google Colab

[TODO]

Most notebooks require external data files and `.py` files, so you cannot simply import them by themselves.
