import json
import copy
import os
import pandas as pd
import numpy as np
from pytfa.io.json import save_json_model as pytfa_save_json_model
from pytfa.io.json import load_json_model as pytfa_load_json_model
from pytfa.thermo import ThermoModel

# Define the custom attributes as a constant set
CUSTOM_ATTRIBUTES = {'species', 'strain', 'biomass_objective_function'}

def save_custom_json_model(model, filepath):
    """
    Save a ThermoModel object to a JSON file, including specified custom attributes.
    
    :param model: The ThermoModel object.
    :param filepath: Path to the JSON file where the model will be saved.
    """
    # Save the model using pytfa's built-in function
    pytfa_save_json_model(model, filepath)

    # Extract custom attributes and their values
    custom_attrs_dict = {attr: getattr(model, attr, None) for attr in CUSTOM_ATTRIBUTES if hasattr(model, attr)}

    # Append the custom attributes to the JSON file
    with open(filepath, 'r+') as f:
        data = json.load(f)
        data['custom_attributes'] = custom_attrs_dict
        f.seek(0)
        f.truncate()  # Clear the file before writing the updated content
        json.dump(data, f, indent=4)

def load_custom_json_model(filepath):
    """
    Load a ThermoModel object from a JSON file, including specified custom attributes.
    
    :param filepath: Path to the JSON file from which the model will be loaded.
    :return: The ThermoModel object.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Load the model using pytfa's built-in function
    model = pytfa_load_json_model(filepath)

    # If there are custom attributes, set them on the loaded model
    if 'custom_attributes' in data:
        for attr, value in data['custom_attributes'].items():
            setattr(model, attr, value)

    return model

def deepcopy_model(model):
    """
    Creates a deep copy of a ThermoModel object, including custom attributes specified.
    
    :param model: The ThermoModel object to copy.
    :return: A deep copy of the model, including any custom attributes.
    """
    # Create a deep copy of the model using deepcopy
    model_copy = copy.deepcopy(model)
    
    # Explicitly copy each custom attribute if it exists in the original model
    for attr in CUSTOM_ATTRIBUTES:
        if hasattr(model, attr):
            setattr(model_copy, attr, copy.deepcopy(getattr(model, attr)))
    
    return model_copy


def save_results_hdf5(data, file_path, logger, key='data'):
    """
    Save DataFrame results to an HDF5 file.

    Parameters:
        data (DataFrame): The DataFrame to save.
        file_path (str): Full path to the HDF5 file where the data will be saved.
        key (str): The key under which the data is stored in the HDF5 file.
    """
    data.to_hdf(file_path, key=key, mode='w', format='table', data_columns=True)
    logger.info(f"Data saved to {file_path} under key '{key}'")


def load_results_hdf5(file_path, key='data'):
    """
    Load results from an HDF5 file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the HDF5 file.
        key (str): The key under which the data is stored in the HDF5 file.

    Returns:
        DataFrame: Data loaded from the HDF5 file.
    """
    return pd.read_hdf(file_path, key)


import os

def save_results_parquet(data, file_path, logger, index_column_name="index"):
    """
    Save DataFrame results to a Parquet file with the index as a new column.

    Parameters:
        data (DataFrame): The DataFrame to save.
        file_path (str): Full path to the Parquet file where the data will be saved.
        index_column_name (str): Name of the column to store the DataFrame index. Defaults to 'index'.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Add the index as a new column with the specified name
        data = data.reset_index().rename(columns={'index': index_column_name})
        
        # Save DataFrame to Parquet
        data.to_parquet(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise

def load_results_parquet(file_path, logger, index_column_name="index"):
    """
    Load results from a Parquet file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the Parquet file.
        index_column_name (str): Name of the column that was used to store the index.

    Returns:
        DataFrame: Data loaded from the Parquet file.
    """
    try:
        data = pd.read_parquet(file_path)
        logger.info(f"Data loaded from {file_path}")

        # Use the specified column as the index and then drop it
        if index_column_name in data.columns:
            data.set_index(index_column_name, inplace=True)
        
        return data
    except FileNotFoundError as e:
        logger.error(f"File {file_path} does not exist: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise


def process_variability_results(tva_fluxes, logger, tolerance=1e-9):
    """
    Process the results of the variability analysis to ensure that the upper bound
    is always greater than or equal to the lower bound.

    Parameters:
        tva_fluxes (pd.DataFrame): DataFrame containing 'minimum' and 'maximum' columns for reactions.
        tolerance (float): Tolerance value to identify numerical precision issues.

    Returns:
        pd.DataFrame: Processed DataFrame with corrected bounds.
    """
    logger.info("Starting variability results processing.")

    import numpy as np

    # Identify rows where the upper bound is less than the lower bound with tolerance
    invalid_bounds = tva_fluxes[~np.isclose(tva_fluxes['maximum'], tva_fluxes['minimum'], atol=tolerance) & np.less(tva_fluxes['maximum'], tva_fluxes['minimum'])]
    logger.info(f"Found {len(invalid_bounds)} reactions with upper bounds less than lower bounds.")
    tva_fluxes.loc[invalid_bounds.index, 'maximum'] = tva_fluxes.loc[invalid_bounds.index, 'minimum']
    logger.info(f"Corrected {len(invalid_bounds)} reactions with invalid bounds.")

    # Round values to the nearest multiple of the specified tolerance
    tva_fluxes['minimum'] = np.round(tva_fluxes['minimum'] / tolerance) * tolerance #- tolerance
    tva_fluxes['maximum'] = np.round(tva_fluxes['maximum'] / tolerance) * tolerance #+ tolerance
    logger.info("Rounded bounds to the nearest multiple of the specified tolerance.")


    # # Further ensure that bounds are within the specified tolerance
    # close_bounds = tva_fluxes[np.isclose(tva_fluxes['maximum'], tva_fluxes['minimum'], atol=tolerance)]
    # if not close_bounds.empty:
    #     # Adjust close bounds by adding/subtracting tolerance to ensure proper ordering
    #     tva_fluxes.loc[close_bounds.index, 'maximum'] = tva_fluxes.loc[close_bounds.index, 'minimum']
    #     logger.info(f"Adjusted bounds for {len(close_bounds)} reactions to account for numerical tolerance.")
    # else:
    #     logger.info("No reactions found within the specified tolerance.")

    logger.info("Finished variability results processing.")
    return tva_fluxes  

def ensure_directory_exists(directory, logger):
    """
    Ensure that the directory exists. If it doesn't, create it.

    Parameters:
        directory (str): The path to the directory to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")  


def merge_forward_reverse_fluxes(sampling):
    # Extract forward and reverse columns separately
    forward_columns = [col for col in sampling.columns if '_reverse_' not in col]
    reverse_columns = [col for col in sampling.columns if '_reverse_' in col]

    # Dictionary to hold the net flux data
    net_flux_dict = {}

    for forward_col in forward_columns:
        # Find the corresponding reverse column by checking the prefix
        reverse_col = next((col for col in reverse_columns if col.startswith(forward_col + '_reverse')), None)
        
        if reverse_col:
            # Calculate net flux and add to the dictionary
            net_flux_dict[forward_col] = sampling[forward_col] - sampling[reverse_col]
        else:
            # If no reverse column exists, use the forward column as is
            net_flux_dict[forward_col] = sampling[forward_col]

    # Convert the dictionary to a DataFrame
    net_fluxes = pd.DataFrame(net_flux_dict)

    return net_fluxes


def post_process_sampling_results(samples, model, logger):
    # Identify all reaction IDs from the model
    all_reactions = set(reaction.id for reaction in model.reactions)

    # Identify existing reactions in the sampling dataframe
    existing_reactions = set(samples.columns)

    # Find missing reactions that need to be added to the dataframe
    missing_reactions = all_reactions - existing_reactions

    # Add missing reactions to the dataframe with zeros
    for reaction in missing_reactions:
        samples[reaction] = 0.0

    # Remove columns that start with specified prefixes
    samples = samples.loc[:, ~samples.columns.str.startswith(('DG_', 'DGo_', 'LC_'))]

    # Drop the first column (assumed to be unnecessary based on provided code)
    samples.drop(columns=samples.columns[0], inplace=True)

    # Merge forward and reverse fluxes using the provided merge function
    samples = merge_forward_reverse_fluxes(samples)

    return samples
