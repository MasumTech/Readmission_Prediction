import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("diabetic_data.csv")
print(df.shape)


# Calculate the total number of rows (data points) in df
total_data = df.shape[0]
# Print the result
print(f'Total data points: {total_data}')

# Get the total number of unique entities in the 'patient_nbr' column
total_unique_patient_nbr = df['patient_nbr'].nunique()
# Print the total number of unique entities
print(f'Total unique patient_nbr entries: {total_unique_patient_nbr}')

# Remove duplicates based on 'patient_nbr' column and keep only the first occurrence
df.drop_duplicates(subset='patient_nbr', keep='first', inplace=True)
# Print the total number of rows after removing duplicates
print(f'Total rows after removing duplicates from patient_nbr column: {df.shape[0]}')

# # Import the necessary library
# from prettytable import PrettyTable

# # Create a PrettyTable object to display missing data information
# table = PrettyTable(['Column Name', 'Missing Values', 'Missing Percentage'])

# # Iterate through each column in the dataset
# for column in df.columns:

#     # Identify the number of missing values (using '?' as an indicator of missing data)
#     missing_count = len(df[df[column] == '?'])

#     # Compute the percentage of missing values in the column
#     missing_percentage = missing_count / len(df) * 100

#     # Add the column data to the table
#     table.add_row([column, missing_count, f"{missing_percentage:.2f}%"])

# # Display the table
# print(table)

# Drop 'weight' and 'payer_code' columns due to excessive missing values
df = df.drop(['weight', 'payer_code'], axis=1)

# Define conditions for rows to be dropped
conditions = (
    (df['diag_1'] == '?') |
    (df['diag_2'] == '?') |
    (df['diag_3'] == '?') |
    (df['race'] == '?') |
    (df['gender'] == 'Unknown/Invalid')
)

# Drop rows based on the specified conditions
df = df[~conditions].reset_index(drop=True)

# Import the necessary library
import pandas as pd

# Loop through each column in the DataFrame
for col in df.columns:

    # Get unique values for the current column
    unique_vals = df[col].unique()

    # Print column name and unique values or count based on threshold
    if unique_vals.size < 30:
        print(f"{col}:")
        print(unique_vals)
    else:
        print(f"{col}: {unique_vals.size} unique values")

# Drop identifiers that do not contribute to the analysis
df = df.drop(columns=['encounter_id', 'patient_nbr'])

# Drop attributes with constant values
df = df.drop(columns=['citoglipton', 'examide', 'glimepiride-pioglitazone', 'metformin-rosiglitazone'])

# Filter out patient samples associated with discharge dispositions indicating death or hospice care
df = df.loc[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]

# Display unique values in the 'age' column
print(np.unique(df['age']))

# Create a dictionary to map age ranges to their mean values
age_mapping = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}

# Map the age ranges to their mean values using the age_mapping dictionary
df['age'] = df['age'].map(age_mapping)

# Display the first 5 rows of the transformed 'age' column
print(df['age'].head())

# Reclassify the 'discharge_disposition_id' based on predefined categories
df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(
    lambda x: 1 if int(x) in [6, 8, 9, 13] else
              (2 if int(x) in [3, 4, 5, 14, 22, 23, 24] else
              (10 if int(x) in [12, 15, 16, 17] else
              (11 if int(x) in [19, 20, 21] else
              (18 if int(x) in [25, 26] else int(x)))))
)

# Remove rows with certain 'discharge_disposition_id' values
df = df[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]

# Reclassify the 'admission_type_id' into fewer categories
df['admission_type_id'] = df['admission_type_id'].apply(
    lambda x: 1 if int(x) in [2, 7] else
              (5 if int(x) in [6, 8] else int(x))
)

# Reclassify the 'admission_source_id' into fewer categories
df['admission_source_id'] = df['admission_source_id'].apply(
    lambda x: 1 if int(x) in [2, 3] else
              (4 if int(x) in [5, 6, 10, 22, 25] else
              (9 if int(x) in [15, 17, 20, 21] else
              (11 if int(x) in [13, 14] else int(x))))
)

# Pre-defined specialty categories
high_frequency = [
    'InternalMedicine', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
    'Orthopedics', 'Orthopedics-Reconstructive', 'Emergency/Trauma', 'Urology',
    'ObstetricsandGynecology', 'Psychiatry', 'Pulmonology', 'Nephrology', 'Radiologist'
]

low_frequency = [
    'Surgery-PlasticwithinHeadandNeck', 'Psychiatry-Addictive', 'Proctology',
    'Dermatology', 'SportsMedicine', 'Speech', 'Perinatology', 'Neurophysiology',
    'Resident', 'Pediatrics-Hematology-Oncology', 'Pediatrics-EmergencyMedicine',
    'Dentistry', 'DCPTEAM', 'Psychiatry-Child/Adolescent', 'Pediatrics-Pulmonology',
    'Surgery-Pediatric', 'AllergyandImmunology', 'Pediatrics-Neurology',
    'Anesthesiology', 'Pathology', 'Cardiology-Pediatric', 'Endocrinology-Metabolism',
    'PhysicianNotFound', 'Surgery-Colon&Rectal', 'OutreachServices',
    'Surgery-Maxillofacial', 'Rheumatology', 'Anesthesiology-Pediatric',
    'Obstetrics', 'Obsterics&Gynecology-GynecologicOnco'
]

pediatrics = [
    'Pediatrics', 'Pediatrics-CriticalCare', 'Pediatrics-EmergencyMedicine',
    'Pediatrics-Endocrinology', 'Pediatrics-Hematology-Oncology', 'Pediatrics-Neurology',
    'Pediatrics-Pulmonology', 'Anesthesiology-Pediatric', 'Cardiology-Pediatric',
    'Surgery-Pediatric'
]

psychic = [
    'Psychiatry-Addictive', 'Psychology', 'Psychiatry', 'Psychiatry-Child/Adolescent',
    'PhysicalMedicineandRehabilitation', 'Osteopath'
]

neurology = [
    'Neurology', 'Surgery-Neuro', 'Pediatrics-Neurology', 'Neurophysiology'
]

surgery = [
    'Surgeon', 'Surgery-Cardiovascular', 'Surgery-Cardiovascular/Thoracic',
    'Surgery-Colon&Rectal', 'Surgery-General', 'Surgery-Maxillofacial',
    'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck', 'Surgery-Thoracic',
    'Surgery-Vascular', 'SurgicalSpecialty', 'Podiatry'
]

ungrouped = [
    'Endocrinology', 'Gastroenterology', 'Gynecology', 'Hematology',
    'Hematology/Oncology', 'Hospitalist', 'InfectiousDiseases', 'Oncology',
    'Ophthalmology', 'Otolaryngology', 'Pulmonology', 'Radiology'
]

missing = ['?']

def categorize_specialty(value):
    if value in pediatrics:
        return 'pediatrics'
    elif value in psychic:
        return 'psychic'
    elif value in neurology:
        return 'neurology'
    elif value in surgery:
        return 'surgery'
    elif value in high_frequency:
        return 'high_freq'
    elif value in low_frequency:
        return 'low_freq'
    elif value in ungrouped:
        return 'ungrouped'
    elif value in missing:
        return 'missing'
    else:
        return value  # Retain the original value if not categorized

# Apply the categorization function to the 'medical_specialty' column
df['medical_specialty'] = df['medical_specialty'].apply(categorize_specialty)


def categorize_diagnosis(row, icd9):
    """
    Categorizes diagnosis codes based on ICD-9 standards.
    Input diagnosis codes are grouped into more general categories.
    """
    code = row[icd9]

    if code.startswith(("E", "V")):
        return "Other"
    else:
        num = float(code)

        if 390 <= num <= 459 or num == 785:
            return "circulatory"
        elif 520 <= num <= 579 or num == 787:
            return "digestive"
        elif 580 <= num <= 629 or num == 788:
            return "genitourinary"
        elif np.trunc(num) == 250:
            return "diabetes"
        elif 800 <= num <= 999:
            return "injury"
        elif 710 <= num <= 739:
            return "musculoskeletal"
        elif 140 <= num <= 239:
            return "neoplasms"
        elif 460 <= num <= 519 or num == 786:
            return "respiratory"
        else:
            return "other"

# Apply the categorization function to the 'diag_1', 'diag_2', and 'diag_3' columns
df["diag1_norm"] = df.apply(categorize_diagnosis, axis=1, icd9="diag_1")
df["diag2_norm"] = df.apply(categorize_diagnosis, axis=1, icd9="diag_2")
df["diag3_norm"] = df.apply(categorize_diagnosis, axis=1, icd9="diag_3")

# Drop the original diagnosis columns from the DataFrame
df.drop(columns=['diag_1', 'diag_2', 'diag_3'], inplace=True)

# Create a new feature for service utilization
df['patient_service'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# List of medication keys
medication_keys = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
    'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide'
]

# Recoding medication use into binary variables and storing them in new columns
for medication in medication_keys:
    new_col_name = f"{medication}_new"
    df[new_col_name] = df[medication].apply(lambda x: 0 if x in ['No', 'Steady'] else 1)

# Initialize 'med_change' column to 0
df['med_change'] = 0

# Summing up the new columns to create the 'med_change' feature
for medication in medication_keys:
    new_col_name = f"{medication}_new"
    df['med_change'] += df[new_col_name]  # Add to 'med_change'
    df.drop(columns=new_col_name, inplace=True)  # Remove the intermediate new columns

# Display the distribution of the 'med_change' feature
print(df['med_change'].value_counts())

# List of medication keys
keys = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
    'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide'
]

# Convert medication indications to binary values (1 for used, 0 for not used)
df[keys] = df[keys].replace({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1})

# Calculate total number of medications used for each patient
df['num_med'] = df[keys].sum(axis=1)

import numpy as np
import pandas as pd

# Define the list of numerical columns
num_col = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
           'num_medications', 'number_outpatient', 'number_emergency',
           'number_inpatient', 'number_diagnoses']

# Initialize a new DataFrame to store statistics
statdataframe = pd.DataFrame()
statdataframe['numeric_column'] = num_col

# Initialize lists to store the statistics
skew_before = []
skew_after = []
kurt_before = []
kurt_after = []
standard_deviation_before = []
standard_deviation_after = []
log_transform_needed = []
log_type = []

# For each column in the list of numerical columns
for col in num_col:
    # Compute skewness, kurtosis, and standard deviation before transformation
    skewval = df[col].skew()
    kurtval = df[col].kurtosis()
    sdval = df[col].std()

    # Store the before statistics
    skew_before.append(skewval)
    kurt_before.append(kurtval)
    standard_deviation_before.append(sdval)

    # Check if transformation is needed
    if (abs(skewval) > 2) & (abs(kurtval) > 2):
        log_transform_needed.append('Yes')

        # Calculate the proportion of zero values in the column
        zero_proportion = (df[col] == 0).mean()

        # Apply appropriate transformation based on the proportion of zeros
        if zero_proportion < 0.02:
            log_type.append('log')
            transformed_data = np.log(df[col].replace(0, np.nan))
        else:
            log_type.append('log1p')
            transformed_data = np.log1p(df[col])

        # Compute statistics after transformation
        skew_after_val = transformed_data.skew()
        kurt_after_val = transformed_data.kurtosis()
        sd_after_val = transformed_data.std()

        # Store the after statistics
        skew_after.append(skew_after_val)
        kurt_after.append(kurt_after_val)
        standard_deviation_after.append(sd_after_val)
    else:
        log_transform_needed.append('No')
        log_type.append('NA')

        # Store the original statistics for 'after' as no transformation was needed
        skew_after.append(skewval)
        kurt_after.append(kurtval)
        standard_deviation_after.append(sdval)

# Add all computed statistics to the DataFrame
statdataframe['skew_before'] = skew_before
statdataframe['kurtosis_before'] = kurt_before
statdataframe['standard_deviation_before'] = standard_deviation_before
statdataframe['log_transform_needed'] = log_transform_needed
statdataframe['log_type'] = log_type
statdataframe['skew_after'] = skew_after
statdataframe['kurtosis_after'] = kurt_after
statdataframe['standard_deviation_after'] = standard_deviation_after

# Print the DataFrame
statdataframe

# Define a function to apply log transformations based on the transformation type
def apply_log_transformation(df, column, log_type):
    if log_type == 'log':
        df = df[df[column] > 0]  # Keep only positive values
        df[column + "_log"] = np.log(df[column])
    elif log_type == 'log1p':
        df[column + "_log1p"] = np.log1p(df[column])  # Log1p handles zero values
    return df

# Loop through the statdataframe to apply the transformations
for _, row in statdataframe.iterrows():
    if row['log_transform_needed'] == 'Yes':
        colname = row['numeric_column']
        log_type = row['log_type']

        # Apply the appropriate transformation based on log_type
        df = apply_log_transformation(df, colname, log_type)

# Drop the original columns that are not needed anymore
df.drop(columns=['number_outpatient', 'number_inpatient', 'number_emergency'], inplace=True)

# Output the shape and preview the updated DataFrame
print(df.shape)
df.head()

# Select the numeric columns
num_cols = ['age', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_diagnoses']

# Calculate the mean and standard deviation for each numeric column
means = df[num_cols].mean()
stds = df[num_cols].std()

# Identify the rows where the values are within 3 standard deviations for each numeric column
df_filtered = df[(np.abs((df[num_cols] - means) / stds) < 3).all(axis=1)]

# Print the updated shape and head of the dataframe
print(df_filtered.shape)
df_filtered.head()

# Convert 'readmitted' to a binary format (1 for '<30', 0 for '>30' or 'No')
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

df.drop('acetohexamide', axis=1, inplace=True)

# # Convert 'race' column into dummy/indicator variables
# df = pd.get_dummies(df, columns = ["race"], prefix = "race", drop_first=True)
# # Apply one-hot encoding to 'gender' column
# df = pd.get_dummies(df, columns=['gender'], prefix = "gender", drop_first=True)

def diagnose(df, diag):
    if (df["diag1_norm"] == diag) | (df["diag2_norm"] == diag) | (df["diag3_norm"] == diag):
        return True
    else:
        return False

# Check for the presence of certain diagnoses and create new columns for each diagnosis
for val in ['diabetes', 'other', 'circulatory', 'neoplasms', 'respiratory', 'injury', 'musculoskeletal', 'digestive', 'genitourinary']:
    name = val + "_diagnosis"
    df[name] = df.apply(diagnose, axis = 1, diag=val).astype(int)
# Define the columns to be dropped
dropped_Cols=['diag1_norm', 'diag2_norm', 'diag3_norm']

# Drop the defined columns from the DataFrame 'df'
df.drop(columns=dropped_Cols, inplace=True)

common_drugs = ['metformin', 'repaglinide', 'glimepiride', 'glipizide',
                'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
rare_drugs = ["nateglinide", "chlorpropamide", "tolbutamide",
             "acarbose", "miglitol", "troglitazone", "tolazamide",
             "glyburide-metformin", "glipizide-metformin",
               "metformin-pioglitazone"]

# Combine common and rare drugs into a single list
drugs = common_drugs + rare_drugs

# Create a binary indicator for each drug in the list using apply and a lambda function
df_binary = df[drugs].apply(lambda x: x.isin(["Down", "Steady", "Up"]).astype(int))

# Rename the binary columns to indicate drug consumption
df_binary.columns = ["take_" + drug for drug in drugs]

# Concatenate the binary DataFrame with the original DataFrame (excluding the original drug columns)
df = pd.concat([df.drop(columns=drugs), df_binary], axis=1)

import pandas as pd

# Check if 'A1Cresult' column exists before applying get_dummies
if 'A1Cresult' in df.columns:
    # Apply get_dummies to 'A1Cresult' column
    df = pd.get_dummies(df, columns=['A1Cresult'], drop_first=False)

    # Safely drop 'A1Cresult_None' if it exists
    if 'A1Cresult_None' in df.columns:
        df = df.drop(["A1Cresult_None"], axis=1)
    else:
        print("'A1Cresult_None' column not found in DataFrame.")
else:
    print("'A1Cresult' column not found in DataFrame.")


import pandas as pd

# Check if 'max_glu_serum' exists before applying get_dummies
if 'max_glu_serum' in df.columns:
    # Apply pd.get_dummies to the 'max_glu_serum' column
    df = pd.get_dummies(df, columns=['max_glu_serum'], drop_first=False)

    # Check if 'max_glu_serum_None' exists before trying to drop it
    if 'max_glu_serum_None' in df.columns:
        df = df.drop(["max_glu_serum_None"], axis=1)
        print("'max_glu_serum_None' column has been dropped.")
    else:
        print("'max_glu_serum_None' column not found in DataFrame.")
else:
    print("'max_glu_serum' column not found in DataFrame.")

# Update the 'change' column to boolean values
df.loc[df.change == "Ch", "change"] = True
df.loc[df.change == "No", "change"] = False
df['change'] = df['change'].astype(int)  # Convert boolean values to integers (0 or 1)

# Update the 'diabetesMed' column to boolean values
df.loc[df.diabetesMed == "Yes", "diabetesMed"] = True
df.loc[df.diabetesMed == "No", "diabetesMed"] = False
df['diabetesMed'] = df['diabetesMed'].astype(int)  # Convert boolean values to integers (0 or 1)

# One-hot encode the 'medical_specialty' column and drop the first column
df = pd.get_dummies(df, columns=['medical_specialty'], prefix=['med_spec'], drop_first=True)

df = df.drop(["med_spec_missing"], axis = 1)

target_counts = df['readmitted'].value_counts()
print(target_counts)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Assuming df is your preprocessed DataFrame

# Select the 10 features for input
X = df[['discharge_disposition_id', 'age', 'time_in_hospital', 'number_diagnoses', 
        'num_procedures', 'race', 'circulatory_diagnosis', 'number_inpatient_log1p', 
        'admission_type_id', 'num_medications']]

y = df['readmitted']  # Target variable (adjust this to your target column)

# Convert categorical variables to numeric (if necessary)
X = pd.get_dummies(X)  # This encodes categorical variables, adjust as needed

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('Gradient_Boosting_diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as 'Gradient_Boosting_diabetes_model.pkl'")
