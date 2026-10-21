# AISE 4010 Project — Group 11

Forecasting household electricity consumption (`Global_active_power`, kW) with deep learning models.

## Dataset

**Individual Household Electric Power Consumption** (UCI Machine Learning Repository, dataset #235)

- Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- One household in Sceaux, France. 1-minute samples from Dec 2006 to Nov 2010 (about 2.07M rows, 9 columns).
- License: CC BY 4.0

The data files are not committed (they are about 130 MB each). To get them:

1. Download the zip from the link above.
2. Extract `household_power_consumption.txt` into the repository root, next to the notebook:

```
AISE4010_Project_Group11.ipynb
household_power_consumption.txt   <- place here
```

The first cell of the notebook converts the `.txt` file to `household_power_consumption.csv` (only if the CSV does not exist yet).

## Setup

```
conda env create -f 4010-DeepLearning.yaml
conda activate 4010-DeepLearning
```

Then open `AISE4010_Project_Group11.ipynb` and run all cells top to bottom.
