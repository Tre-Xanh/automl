import os
import asyncio
import httpx
from loguru import logger
import pandas as pd


async def async_request_api(
    df: pd.DataFrame,
    client: httpx.AsyncClient,
    scoring_uri: str =os.getenv("SCORING_URI", "http://127.0.0.1:5000/invocations"),
    ):
    scoring_uri = scoring_uri.strip("'\"")
    logger.info(f"start {scoring_uri} ...")
    data = df.to_json(orient="split", index=False)
    res = await client.post(
        scoring_uri, data=data, headers={"Content-type": "application/json"},
    )
    logger.debug(res)
    res_js = res.json()
    # logger.debug(res_js)
    res_df = pd.DataFrame(res_js)
    logger.debug(res_df)
    logger.info(f"... done {scoring_uri}")
    return res_df

def request_api(**kargs):
    client = httpx.AsyncClient()
    kargs = {**kargs, "client": client}
    return asyncio.run(async_request_api(**kargs))