import os
import dotenv

exists = False
config = {}


def get_configs():

    if exists:
        return config

    try:
        config.update(dotenv.main.DotEnv(os.environ.get("DOTENV_PATH", dotenv.find_dotenv())).dict())
    except Exception:
        raise

    return config
