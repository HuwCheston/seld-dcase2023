# Extracts the features, labels, and normalizes the development and evaluation split features.
from argparse import ArgumentParser
from pathlib import Path

import cls_feature_class
import parameters


def main(task_id, data_dir):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    # use parameter set defined by user
    params = parameters.get_params(task_id)

    # allow override of parameters
    if data_dir:
        data_dir = Path(data_dir).resolve()
        params["dataset_dir"] = data_dir
        params["feat_label_dir"] = data_dir / "feat_label"
        params["model_dir"] = data_dir / "models"
        params["dcase_output_dir"] = data_dir / "results"

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    # # Extract features and normalize them
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # # Extract labels
    dev_feat_cls.extract_all_labels()


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True, default="3", )
    ap.add_argument("--data-dir", type=str, required=False, default=None, help="Use this to override the data directory in parameters.py")

    args = vars(ap.parse_args())

    main(**args)
