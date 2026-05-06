# production-ready-ml-pipeline
A robust machine learning workflow that emphasised engineering and reusability.

### General overview
This project integrated a pipeline that combined high-dimensional genomic data with environmental timestamp data 
to predict the yield of milo plants. The project used a Gaussian Process Regression (GPR) model as a proxy for GBLUP to capture a composite three-way kernel that combined genetic, environmental, and genotype by environmental interaction signals for yield prediction.


**Set up and run instructions**

Project Structure
```
├── config.yaml                 # custom setup for pipeline configuration and specifications 
├── main.py                     # point of entry to pipeline
├── README.md                   # project overview, set up, and run instructions
├── src/                        # source code dir
├── ├── eda.py                  # contains plotting logic for initial EDA and final model evaluation
│   ├── processor.py            # handles quality checks, missing data, snps recoding, and so on
│   ├── feature_engineering.py  # engineers features and prepares data for modeling
├── ├── modeling.py             # contains the modeling logic for GPR implementation. Builds and saves a model
│   └── predict.py              # loads existing model to predict new individuals
├── data/                       # data directory 
│   ├── raw/                    # raw data
│   └── processed/              # encoded/refined features and metadata
└── outputs/                    # result dir
    ├── plots/                  # initial EDA and prediction scatter plots
    ├── metrics/                # metrics.json, contains metrics for CV and external test data. rMP: predictive ability,                                         RMSE: root mean squared error, bias.
    └── models/                 # trained model .pkl file
    
```

**Set up instructions**
1. Clone Repository
```bash
git clone https://github.com/olu-rgb/production-ready-ml-pipeline.git
cd cucumber_repo
```
2. Set up a virtual environment to manage dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate # for mac
.venv\Scripts\activate.bat # windows cmd
```
3. Install the required packages using the requirements.txt file
```bash
pip install -r requirements.txt
```
4. Edit config.yaml to set custom specifications.
5. Run the full data processing, modeling, and prediction workflow.
```bash
python main.py
```
6. View results:
  - Metrics are saved in 'outputs/metrics'
      - rMP (predictive ability): is the pearson correlation between the true and predicted scores. Higher values are desired.
      - RMSE (root mean squared error): margin of error of the model, lower values are desired.
      - Bias: this is the slope bias, and it describes model calibration. While a value close to 1 is desired (no bias), <1 is overestimating 
        predicted values, and >1 is underestimating predicted values.
  - If enabled, plots are saved in 'outputs/plots'
