import os
import requests
from loguru import logger
import pandas as pd

def request_api(
    df: pd.DataFrame,
    scoring_uri=os.getenv("SCORING_URI", "http://127.0.0.1:5000/invocations")
    ):
    scoring_uri = scoring_uri.strip("'\"")
    # logger.debug(f"scoring_uri {scoring_uri}")
    data = df.to_json(orient="split", index=False)
    res = requests.post(
        scoring_uri, data=data, headers={"Content-type": "application/json"},
    )
    res_js = res.json()
    # logger.debug(res_js)
    res_df = pd.DataFrame(res_js)
    logger.debug(res_df)
    return res_df
