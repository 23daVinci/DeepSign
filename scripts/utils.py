import yaml


def load_config(path="../config.yaml"):
    """
    Load configuration from a YAML file.
    
    args:
        path (str): Path to the YAML configuration file.
    
    returns:
        dict: Configuration parameters as a dictionary.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config