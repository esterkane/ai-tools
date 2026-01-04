def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_to_file(data, file_path):
    import json
    with open(file_path, 'w') as file:
        json.dump(data, file)

def load_json(file_path):
    import json
    with open(file_path, 'r') as file:
        return json.load(file)

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)