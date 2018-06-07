from ConfigParser import ConfigParser

def get_config(config_path="/home/wl/AliScene/project.conf"):
    config = ConfigParser()
    config.read(config_path)
    return config