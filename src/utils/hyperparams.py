import yaml

def load_hyperparams(yml_file_path):
    """
    Loads hyperparameters from a .yml file.

    Args:
        yml_file_path (str): Path to the .yml file.

    Returns:
        dict: Dictionary containing the hyperparameters.
    """
    
    try:
        with open(yml_file_path, 'r') as file:
            hyperparams = yaml.safe_load(file)
        return hyperparams
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {yml_file_path} does not exist.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")