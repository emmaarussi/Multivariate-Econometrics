######CODE FOR MVE DOWN THE WATER ######
###BY
###Niels van Herk
###Tess Scholtus
###Emma Arussi
###Marta Chejduk
### GROUP 44

#PART 2
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron
import ruptures as rpt
from arch.unitroot import ZivotAndrews
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.api import OLS, add_constant
from arch.unitroot.cointegration import FullyModifiedOLS
import statsmodels.tsa.stattools as adf
from statsmodels.tsa.stattools import coint
from concurrent.futures import ProcessPoolExecutor


#import data
df = pd.read_csv("/Users/tessscholtus/Downloads/VU_MultivariateEcnmtrcs_assignment_dataset.csv")
df.head()

# Filter data for Germany and France
germany_data = df[df['Name'] == 'Germany']
france_data = df[df['Name'] == 'France']

# List of variables to analyze
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

# Summary statistics
print("Summary statistics for Germany:")
print(germany_data[variables].describe())
print("\nSummary statistics for France:")
print(france_data[variables].describe())

# Calculate summary statistics
germany_summary = germany_data[variables].describe()
france_summary = france_data[variables].describe()

# Combine into a single DataFrame with multi-level columns
summary_stats = pd.concat([germany_summary, france_summary], axis=1, keys=['Germany', 'France'])

# Export as LaTeX table for Overleaf
latex_code = summary_stats.to_latex(column_format="lcccccccc", float_format="%.2f", bold_rows=True, caption="Summary Statistics for Germany and France", label="tab:summary_statistics")

# Print the LaTeX code (for preview or direct copy)
print(latex_code)

# Filter data for Germany and France
germany_data = df[df['Name'] == 'Germany']
france_data = df[df['Name'] == 'France']

# List of variables to analyze
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

# Creating the 7x2 grid of plots for comparison
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(16, 20))
fig.suptitle('Time Series Analysis for countries: Germany and France', fontsize=18, weight='bold')

for i, variable in enumerate(variables):
    # Plotting for Germany
    sns.lineplot(data=germany_data, x='year', y=variable, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'{variable} over Time (Germany)', fontsize=12, weight='bold')
    axes[i, 0].set_xlabel('Year')
    axes[i, 0].set_ylabel(variable)

    # Plotting for France
    sns.lineplot(data=france_data, x='year', y=variable, ax=axes[i, 1], color='green')
    axes[i, 1].set_title(f'{variable} over Time (France)', fontsize=12, weight='bold')
    axes[i, 1].set_xlabel('Year')
    axes[i, 1].set_ylabel(variable)

# Adjust layout and aesthetics
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves space for the main title
plt.show()

# Filter data for Germany and France
germany_data = df[df['Name'] == 'Germany'].copy()
france_data = df[df['Name'] == 'France'].copy()

# List of variables to analyze
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

# Perform transformations directly for each variable
for variable in variables:
    # 1. Log transformation
    germany_data[f'{variable}_log'] = np.log(germany_data[variable].clip(lower=1e-10))
    france_data[f'{variable}_log'] = np.log(france_data[variable].clip(lower=1e-10))

    # 2. First differences
    germany_data[f'{variable}_diff'] = germany_data[variable].diff()
    france_data[f'{variable}_diff'] = france_data[variable].diff()


# Define a function to create subplots for a specific transformation
def plot_transformed_data(data_germany, data_france, variables, transformation, title, ylabel):
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(16, 20))
    fig.suptitle(title, fontsize=18, weight='bold')

    for i, variable in enumerate(variables):
        # Plot for Germany
        sns.lineplot(data=data_germany, x='year', y=f'{variable}{transformation}', ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f'{variable} {ylabel} (Germany)', fontsize=12, weight='bold')
        axes[i, 0].set_xlabel('Year')
        axes[i, 0].set_ylabel(ylabel)

        # Plot for France
        sns.lineplot(data=data_france, x='year', y=f'{variable}{transformation}', ax=axes[i, 1], color='green')
        axes[i, 1].set_title(f'{variable} {ylabel} (France)', fontsize=12, weight='bold')
        axes[i, 1].set_xlabel('Year')
        axes[i, 1].set_ylabel(ylabel)

    # Adjust layout and aesthetics
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Plot the log transformations
plot_transformed_data(
    germany_data, france_data, variables,
    transformation='_log',
    title='Log Transformations for Germany and France',
    ylabel='Log(Value)'
)

# Plot the first differences
plot_transformed_data(
    germany_data, france_data, variables,
    transformation='_diff',
    title='First Differences for Germany and France',
    ylabel='First Difference'
)

###DECOMPOSING
# List of countries and variables
countries = ['Germany', 'France']
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm',
             'kWh_percapita', 'Fertilizer', 'Hydropower']

# Descriptive variable names for better readability in plots
variable_names = {
    'Emission': 'Emissions (tons)',
    'Pop_density': 'Population Density (people/km²)',
    'Agri_percent_land': 'Agricultural Land Area (%)',
    'Cattleperkm': 'Cattle per km²',
    'kWh_percapita': 'Electricity Use (kWh per capita)',
    'Fertilizer': 'Fertilizer Use (kg/ha)',
    'Hydropower': 'Hydropower Generation (%)'
}

# Create a grid layout for the plots
fig, axes = plt.subplots(len(variables), len(countries), figsize=(15, 20), sharex=True)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Add a title to the entire figure
fig.suptitle("Decomposed Time Series: Trend", fontsize=16, y=1.02)


# Counter to track subplot position
plot_index = 0

for variable in variables:
    for country in countries:
        # Filter data for the specific country
        country_data = df[df['Name'] == country].copy()

        # Ensure the 'year' column is in datetime format and set as index
        country_data['year'] = pd.to_datetime(country_data['year'], format='%Y')
        country_data.set_index('year', inplace=True)

        # Sort the data by the datetime index
        country_data.sort_index(inplace=True)

        # Perform seasonal decomposition
        try:
            result = seasonal_decompose(country_data[variable], model='additive', period=12)

            # Extract the Trend component
            trend = result.trend

            # Plot the Trend component in the current subplot
            ax = axes[plot_index]
            ax.plot(trend, color='blue' if country == 'Germany' else 'green', linewidth=1)
            ax.set_title(f"{variable_names[variable]} ({country})", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

            # Set labels only for the bottom row and left column
            if plot_index % len(countries) == 0:
                ax.set_ylabel(variable_names[variable], fontsize=9)
            if plot_index >= len(variables) * (len(countries) - 1):
                ax.set_xlabel("Year", fontsize=9)

        except ValueError as e:
            print(f"Could not decompose {variable} for {country}: {e}")

        # Move to the next subplot
        plot_index += 1

# Adjust layout for better fit
plt.tight_layout()
plt.show()

###Missing var check ###
# Check for missing values in the entire dataset
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Filter data for France and Germany
france_data = df[df['Name'] == 'France']
germany_data = df[df['Name'] == 'Germany']

# Check for missing values in France's data
missing_values_france = france_data.isnull().sum()
print("Missing values in each column for France:")
print(missing_values_france[missing_values_france > 0])

# Check for missing values in Germany's data
missing_values_germany = germany_data.isnull().sum()
print("\nMissing values in each column for Germany:")
print(missing_values_germany[missing_values_germany > 0])


###### PART III ######
# Function to perform KPSS test 4.1
def perform_kpss(series, regression='ct'):
    try:
        kpss_stat, p_value, lags, critical_values = kpss(series, regression=regression, nlags='auto')
        result = {
            "KPSS Statistic": kpss_stat,
            "P-Value": p_value,
            "Lags Used": lags,
            "Critical Values": critical_values
        }
    except Exception as e:
        result = {"Error": str(e)}
    return result

# Perform KPSS test for each variable and each country
results = {}
for country, data in [('Germany', germany_data), ('France', france_data)]:
    for variable in variables:
        print(f"Running KPSS test for {variable} in {country}...")
        results[(country, variable)] = perform_kpss(data[variable].dropna())

# Print KPSS test results and decision
for key, value in results.items():
    country, variable = key
    print(f"KPSS Results for {variable} in {country}:")
    if "Error" in value:
        print(f"Error: {value['Error']}")
    else:
        kpss_stat = value["KPSS Statistic"]
        critical_value_5 = value["Critical Values"]['5%']
        print(f"KPSS Statistic: {kpss_stat}")
        print(f"P-Value: {value['P-Value']}")
        print(f"Lags Used: {value['Lags Used']}")
        print(f"Critical Values: {value['Critical Values']}")

        # Decision at 5% significance level
        if kpss_stat > critical_value_5:
            print("Decision: Reject H0 (The series is not stationary at the 5% level)")
        else:
            print("Decision: Fail to reject H0 (The series is stationary at the 5% level)")
    print("\n")


###DICKY FULLER 4.2
# Function to run simple DF test (without additional lags)
def run_df_test(series, variable_name, country):
    # Run DF test without lagged differences
    df_result = adfuller(series, regression="ct", maxlag=0)
    print(f"\nDickey-Fuller Test for {variable_name} ({country}):")
    print(f"Test Statistic: {df_result[0]}")
    print(f"p-value: {df_result[1]}")
    for key, value in df_result[4].items():
        print(f"Critical Value ({key}): {value}")
    print("Stationary" if df_result[1] < 0.05 else "Non-stationary")

# Run the simple DF test for each variable
for variable in variables:
    # DF test for Germany
    run_df_test(germany_data[variable].dropna(), variable, "Germany")

    # DF test for France
    run_df_test(france_data[variable].dropna(), variable, "France")

###ADF 4.3
# Function to run ADF test with automatic lag selection
def run_adf_test(series, variable_name, country):
    # Run ADF test with automatic lag selection (AIC)
    adf_result = adfuller(series, regression="ct", autolag='AIC')
    print(f"\nAugmented Dickey-Fuller Test for {variable_name} ({country}):")
    print(f"Test Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print(f"# Lags Used: {adf_result[2]}")
    print(f"# Observations Used: {adf_result[3]}")
    for key, value in adf_result[4].items():
        print(f"Critical Value ({key}): {value}")
    print("Stationary" if adf_result[1] < 0.05 else "Non-stationary")

# Run the ADF test for each variable
for variable in variables:
    # ADF test for Germany
    print(f"\nTesting {variable} for Germany:")
    run_adf_test(germany_data[variable].dropna(), variable, "Germany")

    # ADF test for France
    print(f"\nTesting {variable} for France:")
    run_adf_test(france_data[variable].dropna(), variable, "France")

##### Phillips Perron 4.3
# Function to perform PP test and determine order of integration
def pp_test_order_of_integration(series, regression='ct'):
    """
    Perform PP test and determine the order of integration (I(d)).

    Args:
        series (pd.Series): Time series data.
        regression (str): Type of deterministic trend in the test ('c', 'ct', or 'ctt').
                          'c' for constant, 'ct' for constant and trend, 'ctt' for constant, trend, and trend^2.

    Returns:
        dict: Test results including test statistic, p-value, and order of integration.
    """
    results = {}
    current_series = series
    for d in range(3):  # Check levels (d=0), first difference (d=1), second difference (d=2)
        # Perform PP test
        test_result = PhillipsPerron(current_series.dropna(), trend=regression)
        test_stat, p_value = test_result.stat, test_result.pvalue

        results[f"I({d})"] = {
            "Test Statistic": test_stat,
            "p-value": p_value
        }

        # Check if series is stationary
        if p_value < 0.05:
            results["Conclusion"] = f"Stationary at I({d})"
            return results

        # If not stationary, take the difference
        current_series = current_series.diff()

    # If no stationarity found after two differences
    results["Conclusion"] = "Non-stationary or I(2)+"
    return results

# Run PP tests for Germany
for variable in variables:
    print(f"PP Test for {variable} (Germany):")
    results = pp_test_order_of_integration(germany_data[variable])
    for key, value in results.items():
        print(f"{key}: {value}")
    print("\n")

# Run PP tests for France
for variable in variables:
    print(f"PP Test for {variable} (France):")
    results = pp_test_order_of_integration(france_data[variable])
    for key, value in results.items():
        print(f"{key}: {value}")
    print("\n")


###Check for structural breaks
!pip install ruptures
import ruptures as rpt

# Function to detect structural breaks using the PELT algorithm
def detect_structural_breaks(data, variable):
    signal = data[variable].dropna().values  # Drop NaN values and get the variable as a numpy array
    model = "l2"  # Using L2 norm for mean shift
    algo = rpt.Pelt(model=model).fit(signal)
    breakpoints = algo.predict(pen=10)  # Penalty parameter (adjustable)
    return breakpoints

# Detect structural breaks for each variable for Germany and France
results = {"Country": [], "Variable": [], "Breakpoints": [], "Number of Breaks": []}

for country, country_data in [("Germany", germany_data), ("France", france_data)]:
    for variable in variables:
        breakpoints = detect_structural_breaks(country_data, variable)
        results["Country"].append(country)
        results["Variable"].append(variable)
        results["Breakpoints"].append(breakpoints)
        results["Number of Breaks"].append(len(breakpoints) - 1)  # Number of breaks = number of segments - 1

# Convert results to a DataFrame
breaks_df = pd.DataFrame(results)

# Display results
from IPython.display import display
display(breaks_df)

#### Structural breaks
# Function to detect multiple structural breaks using Bai-Perron method
def detect_multiple_breaks(series, variable_name, country):
    algo = rpt.Pelt(model="l2").fit(series.values)
    breakpoints = algo.predict(pen=10)  # Adjust the penalty to control sensitivity
    print(f"\nBai-Perron Test for {variable_name} ({country}):")
    print(f"Detected Break Points: {breakpoints[:-1]}")  # Exclude the last point (end of series)
    # Plot the results
    rpt.display(series.values, breakpoints, figsize=(10, 6))
    plt.title(f"Detected Breaks in {variable_name} ({country})")
    plt.show()
    return breakpoints[:-1]

# Analyze multiple breaks for each variable
for variable in variables:
    print(f"\nAnalyzing {variable} for Germany:")
    germany_series = germany_data[variable].dropna()
    multiple_breaks_germany = detect_multiple_breaks(germany_series, variable, "Germany")

    print(f"\nAnalyzing {variable} for France:")
    france_series = france_data[variable].dropna()
    multiple_breaks_france = detect_multiple_breaks(france_series, variable, "France")


###ZIVOT ANDREWS FOR UNIT ROOTS
def run_zivot_andrews_test(series, variable_name, country):
    """
    Runs the Zivot-Andrews test for a given time series.
    Handles potential issues with data formatting and parameter selection.
    """
    try:
        # Ensure the series is a pandas Series and drop NaN values
        series = series.dropna()

        # Set a reasonable maximum number of lags
        max_lags = min(10, len(series) // 10)

        # Perform the Zivot-Andrews test
        za_result = ZivotAndrews(series, max_lags=max_lags, trend="ct", trim=0.15, method="aic")

        # Print test results
        print(f"\nZivot-Andrews Test for {variable_name} ({country}):")
        print(f"Test Statistic: {za_result.stat}")
        print(f"p-value: {za_result.pvalue}")
        print(f"Break Date: {za_result.breakpoint + 1}")
        for key, value in za_result.critical_values.items():
            print(f"Critical Value ({key}): {value}")
        print("Stationary with Break" if za_result.pvalue < 0.05 else "Non-stationary")

    except Exception as e:
        print(f"Error while running Zivot-Andrews test for {variable_name} ({country}): {e}")


for variable in variables:
    print(f"\nAnalyzing {variable} for Germany:")
    run_zivot_andrews_test(germany_data[variable], variable, "Germany")

    print(f"\nAnalyzing {variable} for France:")
    run_zivot_andrews_test(france_data[variable], variable, "France")


#### PART IV ##### ENGLE AND GRENGER
# Filter data for variables
france_data = france_data[variables]

# Add a time trend to the dataset
france_data['time'] = range(1, len(france_data) + 1)

# Step 1: Multivariate static regression with time trend
# Dependent variable
y = france_data['Emission']
# Independent variables (including time trend)
X = france_data.drop(columns=['Emission'])
X = sm.add_constant(X)  # Add constant for intercept

# Fit the regression model
model = sm.OLS(y, X).fit()

# Step 2: Perform ADF test on the residuals
residuals = model.resid
adf_results = adfuller(residuals, regression='ct')  # 'ct' includes both constant and trend in ADF test

# Extract ADF test statistics
test_statistic = adf_results[0]
p_value = adf_results[1]
critical_values = adf_results[4]

# Print the ADF test results
print("\nStep 2: Augmented Dickey-Fuller Test on Residuals (with Time Trend)")
print(f"ADF Test Statistic: {test_statistic}")
print(f"p-value: {p_value}")
print("Critical Values:")
for key, value in critical_values.items():
    print(f"   {key}: {value}")

# Check if residuals are stationary
if p_value < 0.05:
    print("\nConclusion: Residuals are stationary. Evidence of cointegration for France.")
else:
    print("\nConclusion: Residuals are not stationary. No evidence of cointegration for France.")

# Germany part

# List of variables to analyze
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

# Filter data for variables
germany_data = germany_data[variables]

# Add a time trend to the dataset
germany_data['time'] = range(1, len(germany_data) + 1)

# Step 1: Multivariate static regression with time trend
# Dependent variable
y = germany_data['Emission']
# Independent variables (including time trend)
X = germany_data.drop(columns=['Emission'])
X = sm.add_constant(X)  # Add constant for intercept

# Fit the regression model
model2 = sm.OLS(y, X).fit()

# Step 2: Perform ADF test on the residuals
residuals = model2.resid
adf_results = adfuller(residuals, regression='ct')  # 'ct' includes both constant and trend in ADF test

# Extract ADF test statistics
test_statistic = adf_results[0]
p_value = adf_results[1]
critical_values = adf_results[4]

# Print the ADF test results
print("\nStep 2: Augmented Dickey-Fuller Test on Residuals (with Time Trend)")
print(f"ADF Test Statistic: {test_statistic}")
print(f"p-value: {p_value}")
print("Critical Values:")
for key, value in critical_values.items():
    print(f"   {key}: {value}")

# Check if residuals are stationary
if p_value < 0.05:
    print("\nConclusion: Residuals are stationary. Evidence of cointegration for Germany.")
else:
    print("\nConclusion: Residuals are not stationary. No evidence of cointegration for Germany.")

# MacKinnon Critical Values
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit

# Fetch critical values for Engle-Granger setup (constant + trend, 6 regressors)
critical_values = mackinnoncrit(6, regression="ct")  # 'ct' includes constant and trend

print(critical_values)

# Define a function to perform Phillips-Ouliaris cointegration test
def phillips_ouliaris_test(y, X, critical_values_table, regression='ct'):
    """
    Perform the Phillips-Ouliaris cointegration test for multivariate models.

    Parameters:
        y (array-like): Dependent variable.
        X (DataFrame): Independent variables.
        critical_values_table (dict): Phillips-Ouliaris critical values for different significance levels.
        regression (str): Include 'c' (constant), 'ct' (constant + trend), or 'n' (none).

    Returns:
        Test statistic, p-value, and critical values comparison.
    """
    # Step 1: Multivariate regression
    X = sm.add_constant(X) if regression in ['c', 'ct'] else X  # Add constant if needed
    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # Step 2: Use ADF test on residuals (proxy for Phillips-Ouliaris)
    adf_result = sm.tsa.adfuller(residuals, regression=regression)
    test_stat = adf_result[0]

    # Step 3: Compare test statistic to Phillips-Ouliaris critical values
    crit_values = critical_values_table[len(X.columns) - 1]  # Adjust for multivariate case
    significance = {level: test_stat < crit for level, crit in crit_values.items()}

    return test_stat, adf_result[1], significance, crit_values

# Example setup for France
france_y = france_data['Emission']
france_X = france_data[['Pop_density', 'Agri_percent_land', 'Cattleperkm',
                        'kWh_percapita', 'Fertilizer', 'Hydropower']]

# Example: Predefined critical values (replace with correct table from literature)
# Critical values structure: {number_of_regressors: {significance_level: critical_value}}
po_critical_values = {
    6: {  # 6 regressors (including trend and constant)
        '1%': -4.0,
        '5%': -3.5,
        '10%': -3.2
    }
}

# Perform Phillips-Ouliaris Test
test_stat, p_value, significance, crit_values = phillips_ouliaris_test(
    france_y, france_X, po_critical_values, regression='ct'
)

# Display results
print("Phillips-Ouliaris Test for France")
print(f"Test Statistic: {test_stat}")
print(f"p-value: {p_value}")
print("Critical Values:")
for level, crit_value in crit_values.items():
    print(f"   {level}: {crit_value} ({'Reject Null' if significance[level] else 'Fail to Reject Null'})")

# Repeat for Germany
germany_y = germany_data['Emission']
germany_X = germany_data[['Pop_density', 'Agri_percent_land', 'Cattleperkm',
                          'kWh_percapita', 'Fertilizer', 'Hydropower']]

test_stat_germany, p_value_germany, significance_germany, crit_values_germany = phillips_ouliaris_test(
    germany_y, germany_X, po_critical_values, regression='ct'
)

print("\nPhillips-Ouliaris Test for Germany")
print(f"Test Statistic: {test_stat_germany}")
print(f"p-value: {p_value_germany}")
print("Critical Values:")
for level, crit_value in crit_values_germany.items():
    print(f"   {level}: {crit_value} ({'Reject Null' if significance_germany[level] else 'Fail to Reject Null'})")

# Additional Code for Phillips-Ouliaris Z test statistic
# Set simulation parameters
np.random.seed(42)
n_obs = 100               # Number of observations
k = 6                     # Number of independent variables
num_simulations = 1500    # Number of simulations
critical_levels = [0.01, 0.05, 0.10]  # Critical levels (1%, 5%, 10%)

# Function to generate random walk
def random_walk(n):
    return np.cumsum(np.random.normal(size=n))

# Function to simulate Phillips-Ouliaris Z test statistic
def simulate_phillips_ouliaris_stat(_):
    y = random_walk(n_obs)  # Dependent variable
    X = np.column_stack([random_walk(n_obs) for _ in range(k)])  # Independent variables

    # Add deterministic components: trend and constant
    trend = np.arange(1, n_obs + 1)  # Time trend
    X_with_trend = sm.add_constant(np.column_stack([trend, X]))  # Add constant and trend

    # Perform regression with constant and trend included
    model = sm.OLS(y, X_with_trend).fit()
    residuals = model.resid  # Extract residuals

    # Perform Phillips-Ouliaris Z test on residuals
    result = coint(y, X[:, 0], trend="ct")  # Cointegration test with constant + trend
    return result[0]  # Return the test statistic

# Parallelize the simulation
with ProcessPoolExecutor() as executor:
    test_statistics = list(executor.map(simulate_phillips_ouliaris_stat, range(num_simulations)))

# Calculate critical values
critical_values = np.percentile(test_statistics, [level * 100 for level in critical_levels])

# Display results
print("Critical Values for Phillips-Ouliaris Z Test (Simulated):")
for level, cv in zip(critical_levels, critical_values):
    print(f"{int(level * 100)}% level: {cv}")

# Select only numeric columns for checking
numeric_datag = germany_data.select_dtypes(include=[np.number])

# Define dependent and independent variables
dependent_var = 'Emission'
independent_vars = ['Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

# Extract the dependent variable (Emission)
y = numeric_datag[dependent_var]

# Extract the independent variables
X = numeric_datag[independent_vars]

# Add deterministic components: trend and constant
trend = np.arange(1, len(y) + 1)  # Linear trend
X_with_trend = sm.add_constant(pd.concat([pd.Series(trend, name='Trend', index=y.index), X], axis=1))

# Perform OLS regression
model = sm.OLS(y, X_with_trend).fit()
residuals = model.resid  # Extract residuals

# Perform Phillips-Ouliaris test (Z_t statistic)
result = coint(y, X.iloc[:, 0], trend="ct")  # Cointegration test with constant + trend
po_test_stat = result[0]  # Z_t statistic
p_value = result[1]       # Associated p-value

# Display results
print(f"Phillips-Ouliaris Z_t Test Statistic Germany: {po_test_stat}")

# Select only numeric columns for checking
numeric_dataf = france_data.select_dtypes(include=[np.number])

# Define dependent and independent variables
dependent_var = 'Emission'
independent_vars = ['Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

# Extract the dependent variable (Emission)
y = numeric_dataf[dependent_var]

# Extract the independent variables
X = numeric_dataf[independent_vars]

# Add deterministic components: trend and constant
trend = np.arange(1, len(y) + 1)  # Linear trend
X_with_trend = sm.add_constant(pd.concat([pd.Series(trend, name='Trend', index=y.index), X], axis=1))

# Perform OLS regression
model = sm.OLS(y, X_with_trend).fit()
residuals = model.resid  # Extract residuals

# Perform Phillips-Ouliaris test (Z_t statistic)
result = coint(y, X.iloc[:, 0], trend="ct")  # Cointegration test with constant + trend
po_test_stat = result[0]  # Z_t statistic
p_value = result[1]       # Associated p-value

# Display results
print(f"Phillips-Ouliaris Z_t Test Statistic France: {po_test_stat}")

#ENGLE GRANGER ADF ON RESIDUALS
# List of variables to analyze
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

def perform_multivariate_cointegration_test(data, country_name):
    # Add a time trend to the dataset
    data['time'] = range(1, len(data) + 1)

    # Step 1: Multivariate static regression with time trend
    # Dependent variable
    y = data['Emission']
    # Independent variables (including time trend)
    X = data.drop(columns=['Emission'])
    X = sm.add_constant(X)  # Add constant for intercept

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Step 2: Extract residuals
    residuals = model.resid

    # Step 3: Perform ADF test on residuals (Engle-Granger style)
    adf_results = adfuller(residuals, regression='ct')  # 'ct' includes constant and trend
    adf_test_statistic = adf_results[0]
    adf_p_value = adf_results[1]
    adf_critical_values = adf_results[4]

    # Print Results
    print(f"\nResults for {country_name}:")

    # Engle-Granger Style Results
    print("\nEngle-Granger Test (ADF on Residuals):")
    print(f"Test Statistic: {adf_test_statistic}")
    print(f"p-value: {adf_p_value}")
    print("Critical Values (Engle-Granger Style):")
    for key, value in adf_critical_values.items():
        print(f"   {key}: {value}")

    # Conclusion for Engle-Granger Test
    if adf_p_value < 0.05:
        print(f"\nConclusion (Engle-Granger): Residuals are stationary. Evidence of cointegration.")
    else:
        print(f"\nConclusion (Engle-Granger): Residuals are not stationary. No evidence of cointegration.")

    # Phillips-Ouliaris Approximation
    print("\nNote: True Phillips-Ouliaris test cannot handle multivariate cases directly. This test is approximated by using ADF on residuals.")
    print("For pairwise cointegration, consider running individual coint() tests between the dependent variable and each independent variable.")

# Perform the tests for France
perform_multivariate_cointegration_test(france_data, "France")

# Perform the tests for Germany
perform_multivariate_cointegration_test(germany_data, "Germany")


###JOHANSSEN TRACE TEST FOR GERMANY
T = len(france_data)
max_lags = int(np.floor(T ** (1/3)))
print("Max lags based on T^1/3 rule:", max_lags)

# Filter data
germany_data = germany_data[variables]

# First determine the order of lags by performing by using lag selection criteria: AIC
model = VAR(germany_data)
lag_order_results = model.select_order(maxlags = 2)
print(lag_order_results.summary())


# Perform the Johansen trace test
result_j_test = coint_johansen(germany_data, det_order=-1, k_ar_diff=1) #det_order -> mean / time trend,

# Extract trace statistics and critical values
trace_statistics = result_j_test.lr1
critical_values = result_j_test.cvt

# Create a DataFrame for the results
table = pd.DataFrame({
    "Rank (r*)": range(len(trace_statistics)),
    "Trace Statistic": trace_statistics,
    "Critical Value (95%)": critical_values[:, 1],
    "Decision (H0: r=r* <= k )": np.where(trace_statistics > critical_values[:, 1], "Reject", "Fail to Reject")
})

# Display the table
print(table)

# Extract max eigenvalue statistics and critical values for max eigenvalue test
max_eigen_statistics = result_j_test.lr2
critical_values_max = result_j_test.cvm

# Create a DataFrame for the max eigenvalue test results
max_eigen_table = pd.DataFrame({
    "Rank (r*)": range(len(max_eigen_statistics)),
    "Max Eigenvalue Statistic": max_eigen_statistics,
    "Critical Value (95%)": critical_values_max[:, 1],
    "Decision (H0: r=r* <= k)": np.where(max_eigen_statistics > critical_values_max[:, 1], "Reject", "Fail to Reject")
})

# Display the max eigenvalue test table
print("\nMaximum Eigenvalue Test Results")
print(max_eigen_table)

###JOHANSSEN TRACE TEST FOR FRANCE ####
# Filter data
france_data = france_data[variables]

# First determine the order of lags by performing by using lag selection criteria: AIC
model = VAR(france_data)
lag_order_results = model.select_order(maxlags = 2)
print(lag_order_results.summary())


# Perform the Johansen trace test
result_j_test = coint_johansen(france_data, det_order=1, k_ar_diff=1) #det_order -> mean / time trend,

# Extract trace statistics and critical values
trace_statistics = result_j_test.lr1
critical_values = result_j_test.cvt

# Create a DataFrame for the results
table = pd.DataFrame({
    "Rank (r*)": range(len(trace_statistics)),
    "Trace Statistic": trace_statistics,
    "Critical Value (95%)": critical_values[:, 1],
    "Decision (H0: r=r* <= k )": np.where(trace_statistics > critical_values[:, 1], "Reject", "Fail to Reject")
})

# Display the table
print(table)

# Extract max eigenvalue statistics and critical values for max eigenvalue test
max_eigen_statistics = result_j_test.lr2
critical_values_max = result_j_test.cvm

# Create a DataFrame for the max eigenvalue test results
max_eigen_table = pd.DataFrame({
    "Rank (r*)": range(len(max_eigen_statistics)),
    "Max Eigenvalue Statistic": max_eigen_statistics,
    "Critical Value (95%)": critical_values_max[:, 1],
    "Decision (H0: r=r* <= k)": np.where(max_eigen_statistics > critical_values_max[:, 1], "Reject", "Fail to Reject")
})

# Display the max eigenvalue test table
print("\nMaximum Eigenvalue Test Results")
print(max_eigen_table)


###ARDL BOUND TEST GERMANY AND FRANCE
# Step 1: Prepare the data
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']
france_data = france_data[variables].dropna()

# Define dependent and independent variables
dependent = 'Emission'
independent = [var for var in variables if var != dependent]

y = france_data[dependent]  # Dependent variable
X = france_data[independent]  # Independent variables

# Step 2: Create lagged variables
def create_lagged_variable(data, var, max_lags):
    """
    Create lagged versions of a single variable.
    """
    return pd.concat([data[var].shift(lag) for lag in range(1, max_lags + 1)], axis=1)

def create_lagged_data(y, X, lags_dict):
    """
    Create a DataFrame with lagged variables based on the optimal lags for each variable.
    """
    lagged_data = pd.DataFrame({"Emission": y})
    for var, lags in lags_dict.items():
        lagged_vars = create_lagged_variable(X, var, lags)
        lagged_vars.columns = [f"{var}_lag{i+1}" for i in range(lags)]
        lagged_data = pd.concat([lagged_data, lagged_vars], axis=1)
    return lagged_data.dropna()

# Step 3: Optimize lags for each independent variable
def optimize_lags(y, X, max_lags):
    """
    Optimize lags for each independent variable individually.
    """
    optimal_lags = {}
    for var in X.columns:
        best_aic = float('inf')
        best_lags = 1

        for lags in range(1, max_lags + 1):
            # Create lagged variables for the current variable
            lagged_X = create_lagged_variable(X, var, lags)
            lagged_X.columns = [f"{var}_lag{i+1}" for i in range(lags)]
            data = pd.concat([y, lagged_X], axis=1).dropna()

            # Fit OLS
            X_with_constant = add_constant(data.iloc[:, 1:])
            model = OLS(data.iloc[:, 0], X_with_constant).fit()

            # Update best lags if AIC improves
            if model.aic < best_aic:
                best_aic = model.aic
                best_lags = lags

        optimal_lags[var] = best_lags
    return optimal_lags

# Optimize lags for each variable
max_lags = 3  # Set maximum lags
optimal_lags = optimize_lags(y, X, max_lags)
print("Optimal lags for each independent variable:", optimal_lags)

# Step 4: Create lagged data using optimal lags
lagged_data = create_lagged_data(y, X, optimal_lags)
lagged_X = lagged_data.iloc[:, 1:]  # All columns except the dependent variable
lagged_y = lagged_data.iloc[:, 0]  # Dependent variable

# Step 5: Fit the final OLS model
lagged_X = add_constant(lagged_X)  # Add constant for OLS
final_model = OLS(lagged_y, lagged_X).fit()

# Output the results
print(final_model.summary())


###4 IV STATIC LEAST SQUARES
!pip3 install linearmodels
import statsmodels.api as sm
from statsmodels.tsa.api import adfuller
from linearmodels.panel import PanelOLS


# Filter data for Germany and France
germany_data = df[df['Name'] == 'Germany']
france_data = df[df['Name'] == 'France']


# Selecting the relevant columns for analysis (example)
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']
df_selected = df[variables]
print(df_selected.columns)

# Function to perform static OLS
def static_ols(data, country_name):
    if data.empty:
        print(f"No data available for {country_name}.")
        return

    # Define independent variables (X) and dependent variable (y)
    X = data[['Hydropower', 'Pop_density', 'Agri_percent_land',
              'Cattleperkm', 'kWh_percapita', 'Fertilizer']]
    y = data['Emission']

    # Add a constant to the independent variables for the intercept
    X = sm.add_constant(X)

    # Perform OLS regression
    model = sm.OLS(y, X).fit()

    # Print summary of the results
    print(f"Static OLS Results for {country_name}")
    print(model.summary())

# Run the static OLS for Germany and France
static_ols(germany_data, "Germany")
static_ols(france_data, "France")

####PART 4.4 DOLS -> DECIDE ON LAGS
# Function to add leads and lags of differenced variables
def add_leads_lags(data, var, lags=2, leads=2):
    for lag in range(1, lags + 1):
        data[f"{var}_lag{lag}"] = data[var].shift(lag)
    for lead in range(1, leads + 1):
        data[f"{var}_lead{lead}"] = data[var].shift(-lead)
    return data

# Function to perform DOLS
def dols(data, country_name, lags=3, leads=3):
    if data.empty:
        print(f"No data available for {country_name}.")
        return

    # Independent variables and dependent variable
    independent_vars = ['Hydropower', 'Pop_density', 'Agri_percent_land',
                        'Cattleperkm', 'kWh_percapita', 'Fertilizer']
    dependent_var = 'Emission'

    # Create a copy of the data
    data = data.copy()

    # Add differenced variables and their leads/lags
    for var in independent_vars:
        data[f"{var}_diff"] = data[var].diff()
        data = add_leads_lags(data, f"{var}_diff", lags=lags, leads=leads)

    # Drop rows with NaN values after adding leads/lags
    data = data.dropna()

    # Prepare the regression dataset
    X = data[independent_vars]
    for var in independent_vars:
        X = pd.concat([X, data.filter(like=f"{var}_lag"), data.filter(like=f"{var}_lead")], axis=1)
    y = data[dependent_var]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Perform DOLS regression
    model = sm.OLS(y, X).fit()

    # Print results
    print(f"DOLS Results for {country_name}")
    print(model.summary())

# Run DOLS for Germany and France
dols(germany_data, "Germany")
dols(france_data, "France")

###PART 4. FMOLS
from arch.unitroot.cointegration import FullyModifiedOLS
from scipy.stats import t

# Filter data for Germany and France
germany_data = df[df['Name'] == 'Germany']
france_data = df[df['Name'] == 'France']

# Function to perform FMOLS
def fmols(data, country_name):
    if data.empty:
        print(f"No data available for {country_name}.")
        return

    # Define independent and dependent variables
    independent_vars = ['Hydropower', 'Pop_density', 'Agri_percent_land',
                        'Cattleperkm', 'kWh_percapita', 'Fertilizer']
    dependent_var = 'Emission'

    # Prepare independent variables (X) and dependent variable (y)
    X = data[independent_vars]
    y = data[dependent_var]

    # Perform FMOLS regression
    model = FullyModifiedOLS(y, X)
    result = model.fit()

    # Calculate t-statistics
    t_stats = result.params / result.std_errors

    # Degrees of freedom (df = n - k, where k is the number of parameters including the intercept)
    df_resid = len(y) - len(result.params)

    # Calculate p-values using the t-distribution
    p_values = 2 * (1 - t.cdf(abs(t_stats), df=df_resid))

    # Create results DataFrame
    summary_df = pd.DataFrame({
        "Coefficient": result.params,
        "Std. Error": result.std_errors,
        "t-Statistic": t_stats,
        "P-value": p_values
    })

    # Print results
    print(f"\nFMOLS Results for {country_name}")
    print(summary_df.to_string(index=True))

# Run FMOLS for Germany and France
fmols(germany_data, "Germany")
fmols(france_data, "France")

####### ECM PART IV ####
# Filter data for Germany and France
germany_data = df[df['Name'] == 'Germany'].set_index('year')
france_data = df[df['Name'] == 'France'].set_index('year')

# Ensure both datasets have the same time periods
common_years = germany_data.index.intersection(france_data.index)
germany_data = germany_data.loc[common_years]
france_data = france_data.loc[common_years]

# Select variables of interest
variables = ['Emission', 'Hydropower', 'Pop_density', 'Agri_percent_land',
             'Cattleperkm', 'kWh_percapita', 'Fertilizer']
germany_vars = germany_data[variables]
france_vars = france_data[variables]

# Check stationarity with Augmented Dickey-Fuller Test (ADF)
def check_stationarity(series):
    result = adfuller(series)
    return result[1]  # p-value

print("ADF Test Results:")
for col in variables:
    p_value = check_stationarity(germany_vars[col])
    print(f"ADF p-value for Germany {col}: {p_value}")
    p_value = check_stationarity(france_vars[col])
    print(f"ADF p-value for France {col}: {p_value}")

# Calculate first differences for stationarity
germany_diff = germany_vars.diff().dropna()
france_diff = france_vars.diff().dropna()

# Perform cointegration test (Johansen test)
def johansen_test(df, det_order=-1):
    result = coint_johansen(df, det_order, 3)  # Fixed lags = 3
    return result

cointegration_test_germany = johansen_test(germany_vars)
cointegration_test_france = johansen_test(france_vars)

print("\nJohansen Cointegration Test Results:")
print(f"Germany cointegration test statistics: {cointegration_test_germany.lr1}")
print(f"France cointegration test statistics: {cointegration_test_france.lr1}")

# Calculate the Error Correction Term (ECT)
def calculate_ect(df_level, dependent, independents):
    X = add_constant(df_level[independents])
    y = df_level[dependent]
    long_run_model = OLS(y, X).fit()
    residuals = long_run_model.resid
    return residuals.shift(1)  # Lagged error correction term

# Add ECT to the datasets
germany_vars['ECT'] = calculate_ect(
    germany_vars, 'Emission', ['Hydropower', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer']
)
france_vars['ECT'] = calculate_ect(
    france_vars, 'Emission', ['Hydropower', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer']
)

# Prepare ECM with exactly 3 lags
def run_ecm_with_lags(df_diff, df_level, dependent, independents, lags=3):
    lagged_levels = df_level.shift(1).dropna()
    diffs = df_diff.loc[lagged_levels.index]

    # Add ECT explicitly
    lagged_levels['ECT'] = calculate_ect(df_level, dependent, independents).loc[lagged_levels.index]

    # Include differenced variables and higher-order lags
    X = diffs[independents].copy()
    for lag in range(1, lags + 1):  # Fixed lags = 3
        lagged_diff = df_diff[independents].shift(lag).loc[lagged_levels.index]
        lagged_diff.columns = [f"{col}_lag{lag}" for col in lagged_diff.columns]
        X = pd.concat([X, lagged_diff], axis=1)

    # Include ECT
    X['ECT'] = lagged_levels['ECT']
    X = add_constant(X)  # Add constant

    # Align and drop NaN rows
    X = X.dropna()
    y = diffs[dependent].loc[X.index]

    # Fit the model
    model = OLS(y, X).fit()
    return model

# Run ECM for Germany with fixed 3 lags
germany_ecm = run_ecm_with_lags(
    germany_diff,
    germany_vars,
    'Emission',
    ['Hydropower', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer'],
    lags=3
)
print("\nGermany ECM Summary:")
print(germany_ecm.summary())

# Run ECM for France with fixed 3 lags
france_ecm = run_ecm_with_lags(
    france_diff,
    france_vars,
    'Emission',
    ['Hydropower', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer'],
    lags=3
)
print("\nFrance ECM Summary:")
print(france_ecm.summary())

####### VECM PART ########
######### CORRECT VECM WITHOUT STRUCTURAL BREAKS ##########
import numpy as np
import pandas as pd
from scipy import linalg

def johansen_vecm_estimates(data, p, r):
    """
    Perform Johansen VECM Maximum Likelihood Estimation

    Parameters:
    -----------
    data : pd.DataFrame
        Multivariate time series data
    p : int
        Lag order of the VAR model
    r : int
        Cointegration rank

    Returns:
    --------
    dict with beta (cointegrating vectors) and alpha (adjustment coefficients)
    """
    # Compute first differences and lagged levels
    delta_Y = data.diff().iloc[1:]
    Y_L = data.shift(1).iloc[1:]

    # Prepare lagged differences matrix
    def create_lagged_differences(df, lags):
        lagged_diffs = []
        for lag in range(1, lags + 1):
            lagged_diffs.append(df.shift(lag))
        return pd.concat(lagged_diffs, axis=1).iloc[lags:]

    # Create lagged differences matrix
    delta_Y_L = create_lagged_differences(delta_Y, p)

    # Truncate and align data
    Y_L = Y_L.iloc[p:]
    delta_Y = delta_Y.iloc[p:]

    # Drop NaNs
    delta_Y_L = delta_Y_L.dropna()
    delta_Y = delta_Y.dropna()
    Y_L = Y_L.dropna()

    # Convert to numpy for computations
    delta_Y_np = delta_Y.to_numpy()
    Y_L_np = Y_L.to_numpy()
    delta_Y_L_np = delta_Y_L.to_numpy()

    # Number of observations and variables
    T, m = delta_Y_np.shape

    # Regularization to handle near-singular matrices
    def safe_inv(matrix, reg_factor=1e-8):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    # Compute projection matrix
    def compute_projection_matrix(X):
        XtX_inv = safe_inv(X.T @ X)
        return np.eye(X.shape[0]) - X @ XtX_inv @ X.T

    # Compute residual matrices
    M_delta = compute_projection_matrix(delta_Y_L_np)
    R_0 = M_delta @ delta_Y_np  # Residuals of ΔY on ΔY_L
    R_h = M_delta @ Y_L_np      # Residuals of Y_L on ΔY_L

    # Compute product moment matrices
    S_00 = (1/T) * R_0.T @ R_0
    S_0h = (1/T) * R_0.T @ R_h
    S_h0 = S_0h.T
    S_hh = (1/T) * R_h.T @ R_h

    # Generalized eigenvalue problem
    eigenvalues, eigenvectors = linalg.eig(
        S_h0 @ safe_inv(S_00) @ S_0h,
        S_hh
    )

    # Sort eigenvalues in descending order
    sort_indices = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Select top r eigenvectors
    beta = eigenvectors[:, :r]

    # Normalize beta with regularization
    intermediate_matrix = beta.T @ S_hh @ beta
    regularized_matrix = intermediate_matrix + 1e-8 * np.eye(intermediate_matrix.shape[0])
    regularized_matrix[regularized_matrix < 0] = 1e-8
    beta_norm = beta @ np.linalg.pinv(np.sqrt(regularized_matrix))

    # Compute alpha (adjustment coefficients)
    alpha = -S_0h @ beta_norm

    return {
        'beta': pd.DataFrame(beta_norm,
                             columns=[f'Cointegrating_Vector_{i+1}' for i in range(r)],
                             index=data.columns),
        'alpha': pd.DataFrame(alpha,
                              columns=[f'Cointegrating_Vector_{i+1}' for i in range(r)],
                              index=data.columns)
    }

# Example usage
germany_data = germany_data.dropna()
estimates = johansen_vecm_estimates(germany_data, p=3, r=3)

# Print results
print("Beta (Cointegrating Vectors):")
print(estimates['beta'])
print("\nAlpha (Adjustment Coefficients):")
print(estimates['alpha'])

# Ensure France's data is cleaned
france_data = france_data.dropna()

# Apply the Johansen VECM estimation for France
estimates_france = johansen_vecm_estimates(france_data, p=3, r=3)

# Print the results for France
print("Beta (Cointegrating Vectors) for France:")
print(estimates_france['beta'])
print("\nAlpha (Adjustment Coefficients) for France:")
print(estimates_france['alpha'])

###### VECM WITH STRUCTURAL BREAKS ####
import numpy as np
import pandas as pd
from scipy import linalg

def johansen_vecm_estimates_with_break(data, p, r, break_dummy_col):
    """
    Perform Johansen VECM Maximum Likelihood Estimation with Structural Break

    Parameters:
    -----------
    data : pd.DataFrame
        Multivariate time series data with the break dummy included
    p : int
        Lag order of the VAR model
    r : int
        Cointegration rank
    break_dummy_col : str
        Column name of the break dummy variable

    Returns:
    --------
    dict with beta (cointegrating vectors), alpha (adjustment coefficients), and adjusted data
    """
    # Separate break dummy and main data
    break_dummy = data[[break_dummy_col]].to_numpy()
    main_data = data.drop(columns=[break_dummy_col])

    # Compute first differences and lagged levels
    delta_Y = main_data.diff().iloc[1:]
    Y_L = main_data.shift(1).iloc[1:]
    break_dummy = break_dummy[1:]  # Adjust break dummy for diff

    # Prepare lagged differences matrix
    def create_lagged_differences(df, lags):
        lagged_diffs = []
        for lag in range(1, lags + 1):
            lagged_diffs.append(df.shift(lag))
        return pd.concat(lagged_diffs, axis=1).iloc[lags:]

    delta_Y_L = create_lagged_differences(delta_Y, p)

    # Truncate and align data
    Y_L = Y_L.iloc[p:]
    delta_Y = delta_Y.iloc[p:]
    break_dummy = break_dummy[p:]
    delta_Y_L = delta_Y_L.dropna()

    # Convert to numpy for computations
    delta_Y_np = delta_Y.to_numpy()
    Y_L_np = Y_L.to_numpy()
    delta_Y_L_np = delta_Y_L.to_numpy()

    # Number of observations and variables
    T, m = delta_Y_np.shape

    # Regularization to handle near-singular matrices
    def safe_inv(matrix, reg_factor=1e-8):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    # Compute projection matrix with the break dummy
    X = np.hstack([delta_Y_L_np, break_dummy])  # Include break dummy as exogenous variable
    XtX_inv = safe_inv(X.T @ X)
    M_delta = np.eye(T) - X @ XtX_inv @ X.T

    # Compute residual matrices
    R_0 = M_delta @ delta_Y_np  # Residuals of ΔY on ΔY_L and break dummy
    R_h = M_delta @ Y_L_np      # Residuals of Y_L on ΔY_L and break dummy

    # Compute product moment matrices
    S_00 = (1/T) * R_0.T @ R_0
    S_0h = (1/T) * R_0.T @ R_h
    S_h0 = S_0h.T
    S_hh = (1/T) * R_h.T @ R_h

    # Generalized eigenvalue problem
    eigenvalues, eigenvectors = linalg.eig(
        S_h0 @ safe_inv(S_00) @ S_0h,
        S_hh
    )

    # Sort eigenvalues in descending order
    sort_indices = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Select top r eigenvectors
    beta = eigenvectors[:, :r]

    # Normalize beta with regularization
    intermediate_matrix = beta.T @ S_hh @ beta
    regularized_matrix = intermediate_matrix + 1e-8 * np.eye(intermediate_matrix.shape[0])
    regularized_matrix[regularized_matrix < 0] = 1e-8
    beta_norm = beta @ np.linalg.pinv(np.sqrt(regularized_matrix))

    # Compute alpha (adjustment coefficients)
    alpha = -S_0h @ beta_norm

    return {
        'beta': pd.DataFrame(beta_norm,
                             columns=[f'Cointegrating_Vector_{i+1}' for i in range(r)],
                             index=main_data.columns),
        'alpha': pd.DataFrame(alpha,
                              columns=[f'Cointegrating_Vector_{i+1}' for i in range(r)],
                              index=main_data.columns)
    }

# Prepare data with break dummy for Germany and France
break_year = 2020
germany_data['break_dummy'] = (germany_data.index >= break_year).astype(int)
france_data['break_dummy'] = (france_data.index >= break_year).astype(int)

# Estimate for Germany with structural break
estimates_germany = johansen_vecm_estimates_with_break(germany_data, p=3, r=3, break_dummy_col='break_dummy')
print("Beta (Cointegrating Vectors) for Germany:")
print(estimates_germany['beta'])
print("\nAlpha (Adjustment Coefficients) for Germany:")
print(estimates_germany['alpha'])

# Estimate for France with structural break
estimates_france = johansen_vecm_estimates_with_break(france_data, p=3, r=3, break_dummy_col='break_dummy')
print("Beta (Cointegrating Vectors) for France:")
print(estimates_france['beta'])
print("\nAlpha (Adjustment Coefficients) for France:")
print(estimates_france['alpha'])

 
### PART V ### GRANGER CAUSALITY
###BEST APPROACH FOR GRANGER CAUSALITY MODEL WITHOUT STRUCTURAL BREAKS ######
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# Assume df is a DataFrame with relevant data
# Filter data for Germany and France
germany_data = df[df['Name'] == 'Germany']
france_data = df[df['Name'] == 'France']

# List of variables to analyze
variables = ['Emission', 'Pop_density', 'Agri_percent_land', 'Cattleperkm', 'kWh_percapita', 'Fertilizer', 'Hydropower']

# Filter the relevant columns
germany_endog = germany_data[variables]
france_endog = france_data[variables]

# Manually set lag orders and cointegration ranks
lag_order = 3 #based on VAR part determined with rule of thumb and HCIQ method
coint_rank = 3

# Function to perform VECM and Granger causality test
def perform_vecm_granger(endog, country_name, coint_rank, lag_order):
    vecm_model = VECM(endog=endog, k_ar_diff=lag_order, coint_rank=coint_rank, deterministic="ci")
    vecm_fit = vecm_model.fit()

    # Collect error correction term (ECT) and its t-statistic
    ect_coef = vecm_fit.alpha
    ect_t_stat = vecm_fit.alpha / vecm_fit.stderr_alpha

    # Short-run causality (Wald test) using vecm_fit.test_granger_causality
    results = []
    for caused_idx, caused_var in enumerate(variables):
        short_run_p_values = []
        for causing_idx, causing_var in enumerate(variables):
            if caused_idx != causing_idx:
                test_result = vecm_fit.test_granger_causality(causing_idx, caused_idx, signif=0.05)
                short_run_p_values.append(round(test_result.pvalue, 4))  # Append p-value (rounded to 4 decimals)
            else:
                short_run_p_values.append('-')

        # Collecting long-run causality based on the significance of the error correction term
        long_run_stat = round(ect_t_stat[caused_idx, 0], 4)
        results.append([caused_var] + short_run_p_values + [long_run_stat])

    # Create a summary DataFrame for results
    columns = ['Dependent Variable'] + variables + ['ECT_t-statistic']
    summary_df = pd.DataFrame(results, columns=columns)

    return summary_df

# Perform VECM and Granger causality tests for Germany and France
germany_summary = perform_vecm_granger(germany_endog, "Germany", coint_rank, lag_order)
france_summary = perform_vecm_granger(france_endog, "France", coint_rank, lag_order)

# Display results in the desired table format
print("\nTable: Granger Causality Results for Germany\n")
print(germany_summary)

print("\nTable: Granger Causality Results for France\n")
print(france_summary)

####### VECM WITH STRUCTURAL BREAK IN 2020 #######
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_order

def prepare_country_data(df, country_name, variables):
    """Prepares data for a given country."""
    country_data = df[df['Name'] == country_name]
    country_data = country_data.set_index('year')
    country_data = country_data[variables]
    return country_data

# Prepare data for Germany and France
germany_data = prepare_country_data(df, 'Germany', variables)
france_data = prepare_country_data(df, 'France', variables)

# Ensure data has common years
common_years = germany_data.index.intersection(france_data.index)
germany_data = germany_data.loc[common_years]
france_data = france_data.loc[common_years]

# Add break dummy variables for Germany and France based on Zivot-Andrews results
break_year = 2020

germany_data['break_dummy'] = (germany_data.index >= break_year).astype(int)
france_data['break_dummy'] = (france_data.index >= break_year).astype(int)


# Extract optimal lag order based on HQIC
germany_lags = 3
france_lags = 3
coint_rank = 3

# Function to perform VECM and Granger causality test
def perform_vecm_granger(data, country_name, variables, coint_rank, lag_order):
    vecm_model = VECM(endog=data[variables], k_ar_diff=lag_order, coint_rank=coint_rank, deterministic="ci", exog=data[['break_dummy']])
    vecm_fit = vecm_model.fit()

    # Collect error correction term (ECT) and its t-statistic
    ect_coef = vecm_fit.alpha
    ect_t_stat = vecm_fit.alpha / vecm_fit.stderr_alpha

    # Short-run causality (Wald test) using vecm_fit.test_granger_causality
    results = []
    for caused_idx, caused_var in enumerate(variables):
        short_run_p_values = []
        for causing_idx, causing_var in enumerate(variables):
            if caused_idx != causing_idx:
                test_result = vecm_fit.test_granger_causality(causing_idx, caused_idx, signif=0.05)
                short_run_p_values.append(round(test_result.pvalue, 4))  # Append p-value (rounded to 4 decimals)
            else:
                short_run_p_values.append('-')

        # Collecting long-run causality based on the significance of the error correction term
        long_run_stat = round(ect_t_stat[caused_idx, 0], 4)
        results.append([caused_var] + short_run_p_values + [long_run_stat])

    # Create a summary DataFrame for results
    columns = ['Dependent Variable'] + variables + ['ECT_t-statistic']
    summary_df = pd.DataFrame(results, columns=columns)

    return summary_df

# Perform VECM and Granger causality tests for Germany and France
germany_summary = perform_vecm_granger(germany_data, "Germany", variables, coint_rank=3, lag_order=germany_lags)
france_summary = perform_vecm_granger(france_data, "France", variables, coint_rank=3, lag_order=france_lags)

# Display results in the desired table format
print("\nTable: Granger Causality Results for Germany\n")
print(germany_summary)

print("\nTable: Granger Causality Results for France\n")
print(france_summary)
