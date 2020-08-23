import logging

DATA_URI: str = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("h2o_mlflow")

try:
    import coloredlogs

    coloredlogs.install(level="DEBUG", logger=logger)
except:
    pass
