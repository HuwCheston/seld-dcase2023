"""
Runs evaluation on `dev-test-sony` on a folder of checkpoints.

Can optionally ignore `musicInstrument` class results (not present in data)
"""

import os
from tempfile import TemporaryDirectory
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

import parameters
from cls_compute_seld_results import ComputeSELDResults
from cls_data_generator import DataGenerator
from seldnet_model import SeldModel, MSELoss_ADPIT
from train_seldnet import test_epoch


def proc(model_path: Path, model_idx: int):
    task_id = "3"
    params = parameters.get_params(task_id, do_print=False)

    # Update all parameters with hardcoded filepaths
    dataset_dir = model_path.parent.parent
    params["dataset_dir"] = str(dataset_dir.resolve())
    params["feat_label_dir"] = str((dataset_dir / "feat_label").resolve())
    params["model_dir"] = str((dataset_dir / "models").resolve())
    params["dcase_output_dir"] = str((dataset_dir / "results").resolve())

    # Create test dataset
    print("Getting data...")
    data_gen_test = DataGenerator(
        params=params, split=[5], shuffle=False, per_file=True, do_print=False
    )

    # Define device as whatever is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize evaluation metric class
    score_obj = ComputeSELDResults(params)

    # Create model
    print("Creating model...")
    data_in, data_out = data_gen_test.get_data_sizes()
    model = SeldModel(data_in, data_out, params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = MSELoss_ADPIT()

    # Dump results in DCASE output format for calculating final scores
    with TemporaryDirectory() as dcase_output_test_folder:
        print("Testing...")
        test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)

        test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=True)

        res = []
        for cls_cnt in range(params['unique_classes']):
            res.append(
                dict(
                    cls_cnt=cls_cnt,
                    model_idx=model_idx,
                    ER=classwise_test_scr[0][0][cls_cnt],
                    F=classwise_test_scr[0][1][cls_cnt] * 100,
                    LE=classwise_test_scr[0][2][cls_cnt],
                    LR=classwise_test_scr[0][3][cls_cnt] * 100,
                    SELD=classwise_test_scr[0][4][cls_cnt],
                )
            )

    return res


def main(model_dir, ignore_musicinstrument):
    model_dir = Path(model_dir)
    res = []

    model_idx = 0
    for model_path in os.listdir(model_dir):
        if model_path.endswith(".h5"):
            print(f"Evaluating {model_path}, idx {model_idx}...")
            model_res = proc(model_dir / model_path, model_idx)
            res.extend(model_res)
            model_idx += 1

    df = pd.DataFrame.from_records(res)
    df.round(2).to_csv(model_dir / "averaged_results.csv", index=False)

    if ignore_musicinstrument:
        df = df[df["cls_cnt"] != 9]

    # Print average results overall
    print("Global average results:")
    print(df.groupby("model_idx")[["ER", "F", "LE", "LR", "SELD"]].mean().mean().round(2))

    # groupby class and print class-wise results
    print("Classwise results")
    print(df.groupby("cls_cnt")[["ER", "F", "LE", "LR", "SELD"]].mean().round(2))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--model-dir",
        help="Directory containing .h5 files",
        type=str
    )
    ap.add_argument(
        "--ignore-musicinstrument",
        help="Add this flag to ignore musicInstrument results (not present in test data)",
        action="store_true",
    )
    args = vars(ap.parse_args())
    main(**args)
