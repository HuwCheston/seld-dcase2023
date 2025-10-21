import csv
import os
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch

import cls_feature_class
import parameters
from cls_compute_seld_results import ComputeSELDResults
from cls_data_generator import DataGenerator
from seldnet_model import SeldModel, MSELoss_ADPIT
from train_seldnet import test_epoch


def get_params_feats(model_dir: Path):
    task_id = "3"
    params = parameters.get_params(task_id, do_print=False)

    # Update all parameters with hardcoded filepaths
    params["dataset_dir"] = Path("./LOCATA_dcase").resolve()
    results_path = Path("./LOCATA_dcase/results")
    params["model_dir"] = Path("./LOCATA_dcase/models").resolve()
    params["dcase_output_dir"] = results_path.resolve()

    # Load up feature weights for this dataset
    model_preprocessed_feat_weights = (model_dir.resolve().parent / "feat_label/foa_wts").resolve()
    assert model_preprocessed_feat_weights.is_file()

    lab_path = TemporaryDirectory(dir="LOCATA_dcase")
    params["feat_label_dir"] = lab_path.name

    # Extract features and preprocess based on already extracted weights
    dev_feat_cls = cls_feature_class.FeatureClass(params)
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features(model_preprocessed_feat_weights)
    dev_feat_cls.extract_all_labels()

    return params, lab_path


def proc(model_path: Path, params: dict, thresh: float = 0.5):
    # Create test dataset
    data_gen_test = DataGenerator(
        params=params, split=[1], shuffle=False, per_file=True, do_print=False
    )

    # Define device as whatever is available
    device = "cpu"

    # Initialize evaluation metric class
    score_obj = ComputeSELDResults(params)

    # Create model
    data_in, data_out = data_gen_test.get_data_sizes()
    model = SeldModel(data_in, data_out, params).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    criterion = MSELoss_ADPIT()

    # Dump results in DCASE output format for calculating final scores
    dcase_output_test_folder = os.path.join(params['dcase_output_dir'], "tmp", strftime("%Y%m%d%H%M%S", gmtime()))
    cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
    test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, "cpu", thresh=thresh)

    # Update csv files
    for file in Path(dcase_output_test_folder).rglob(".csv"):
        reader = csv.reader(file, delimiter=',')
        out = []
        for row in reader:
            if row[1] == 1:
                row = [row[0], 0, *row[2:]]
            out.append(row)

        writer = csv.writer(file)
        writer.writerows(out)

    use_jackknife = True
    test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(
        dcase_output_test_folder, is_jackknife=use_jackknife)

    print(f"Results for model path {model_path}")
    print('Class\tER\tF\tLE\tLR\tSELD_score')
    cls_cnt = 0
    print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
        cls_cnt,
        classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                    classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
        classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                    classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
        classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                    classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
        classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                    classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
        classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                    classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))

    return classwise_test_scr[0][2][cls_cnt], classwise_test_scr[0][3][cls_cnt]


def main(model_dir, thresh, ignore_180: bool):
    if isinstance(thresh, float):
        thresh = [thresh]

    model_dir = Path(model_dir)
    params, lab_path = get_params_feats(model_dir)
    res = []
    for th in thresh:
        model_idx = 0
        for model_path in os.listdir(model_dir):
            if model_path.endswith(".h5"):
                le, lr = proc(model_dir / model_path, params, float(th))
                if ignore_180 and le == 180:
                    continue
                res.append(
                    dict(
                        thresh=th,
                        model_idx=model_idx,
                        LE=le,
                        LR=lr * 100,
                    )
                )
                model_idx += 1

    df = pd.DataFrame(res, columns=['model_idx', 'thresh', 'LE', 'LR'])
    df.to_csv(model_dir / "locata_results.csv", index=False)
    lab_path.cleanup()


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--model-dir",
        help="Directory containing .h5 files",
        type=str
    )
    ap.add_argument(
        "--thresh",
        type=float,
        nargs="+",
        help="The threshold values to test"
    )
    ap.add_argument(
        "--ignore-180",
        help="Add this flag to skip over non-predicted classes (as in DCASE-2024)",
        action="store_true",
    )
    args = vars(ap.parse_args())
    main(**args)
