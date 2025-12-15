import copy
import yaml


_cached_config = None
_cached_path = None


def load_config(config_path: str = "wellcounter_config.yml") -> dict:
    """Load the YAML configuration once and return a deep-copied dictionary.

    The configuration is cached after the first read to avoid repeated disk
    access. A deep copy is returned to protect the cached baseline from
    runtime mutation.
    """
    global _cached_config, _cached_path
    if _cached_config is None or _cached_path != config_path:
        with open(config_path, "r") as config_file:
            loaded = yaml.safe_load(config_file)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration file must contain a YAML mapping at the top level.")
        _cached_config = loaded
        _cached_path = config_path
    return copy.deepcopy(_cached_config)
