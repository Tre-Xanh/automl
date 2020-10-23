from pathlib import Path
from loguru import logger

DATA_URI: str = (
    "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
)
Y_TARGET = "Survived"

PRJ_DIR = Path(__file__).parents[1]
DATA_DIR = PRJ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TEST_CSV = str(DATA_DIR / "dftest.csv")