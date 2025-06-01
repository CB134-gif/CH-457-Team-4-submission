import json
import pandas as pd
from pytfa.io.json import load_json_model as pytfa_load_json_model


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

def convert_fluxes_to_sign(net_fluxes):
    # Convert net fluxes to sign representation
    sign_fluxes = net_fluxes.applymap(lambda x: '+1' if x > 0 else ('-1' if x < 0 else '0'))
    return sign_fluxes

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