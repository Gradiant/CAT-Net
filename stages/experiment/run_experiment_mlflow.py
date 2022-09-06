import json
import tempfile
import time

import sys, os
from tools.infer_cls import infer_cls
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')
if path not in sys.path:
    sys.path.insert(0, path)
print(path)
from tools.train import train_model
from tools.infer import infer
from tools.train_DCT_cls import train_model as train_model_cls

from os import path as osp
import fire
import mlflow
from loguru import logger



def log_config_to_mlflow(config, env_info_dict):
    mlflow.log_params(
        {
            "epochs": config.total_epochs,
            "Model": f"{config.model.type}-{config.model.backbone.type}-{config.model.backbone.depth}",
        }
    )

    for k, v in env_info_dict.items():
        if "PyTorch compiling details" in k:
            # Message too long
            continue
        mlflow.log_param(k.replace(",", "-"), v)

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, "config.py"), "w") as f:
            f.write(config.pretty_text)
            f.seek(0)
            mlflow.log_artifact(f.name)


def run_experiment_mlflow():

    logger.info("Start mlflow")
    with mlflow.start_run():

        logger.info("timestamp")

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        artifact_uri = mlflow.get_artifact_uri()[7:]
        logger.info(f'artifact_uri: {artifact_uri}')

        with open("results/experiment.json", "w") as f:
            json.dump({"artifact_uri": artifact_uri, "timestamp": timestamp}, f)

        mlflow.log_artifact("results/experiment.json")

        logger.info("Start")
        train_model()
        # infer(show_mlflow=True)


@logger.catch(reraise=True)
def main():
    fire.Fire(run_experiment_mlflow)


if __name__ == "__main__":
    main()