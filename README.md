# UAV Flight-Path Deviation: ML on Drone Telemetry

Machine-learning pipeline that characterizes how far a hexacopter deviates from its planned flight path, and which onboard and weather factors drive that deviation. The goal is better hyperspectral image quality: a steadier flight path means cleaner imagery.

Research with faculty at the College of Charleston and South Carolina State University. It began as a CSCI 334 course project and continues toward a paper.

## What the pipeline does

1. Parses ArduPilot `.bin` flight logs with `pymavlink` (14 flights, about 31k telemetry samples).
2. Converts GPS to metric coordinates and computes cross-track deviation from the planned path.
3. Joins onboard signals (attitude, tilt, speed) with weather data for each flight.
4. Trains and compares regression and classification models (decision tree, KNN, random forest) with grouped cross-validation, keeping every flight's samples on one side of a split so results are not inflated by within-flight correlation.
5. Runs feature-importance analysis and statistical tests across conditions.

## Repository layout

- `pipeline.py`, `pipeline_v2.py`: end-to-end pipeline (parsing, feature engineering, modeling, evaluation).
- `exploration.ipynb`: exploratory analysis of the telemetry.
- `make_tables.py`: regenerates the result tables used in the report.
- `results/`: metrics for classification and regression, plus feature importance.
- `final_report/`: LaTeX report with figures (deviation histogram, tilt vs. deviation, model comparisons, importance).
- `proposal.pdf`, `midterm_report.pdf`, `CSCi334_project_description.pdf`: project documents.

## Reproducing

```bash
pip install pymavlink pandas numpy scikit-learn scipy matplotlib
python pipeline_v2.py        # writes metrics and importance to results/
python make_tables.py        # regenerates report tables
```

## Tools

Python, pymavlink, pandas, NumPy, scikit-learn, SciPy, Matplotlib, LaTeX.
