import cls_feature_class
import parameters
import sys
import os
import torch
from time import gmtime, strftime
from pathlib import Path
from seldnet_model import SeldModel, MSELoss_ADPIT
from cls_data_generator import DataGenerator
from train_seldnet import test_epoch
from cls_compute_seld_results import ComputeSELDResults


def main(args):
    task_id = "3"
    params = parameters.get_params(task_id)

    model_path, out_name = args

    # Update all parameters with hardcoded filepaths
    params["dataset_dir"] = Path("./LOCATA_dcase").resolve()
    lab_path = Path("./LOCATA_dcase/feat_label").resolve()
    params["feat_label_dir"] = lab_path
    params["model_dir"] = Path("./LOCATA_dcase/models").resolve()
    params["dcase_output_dir"] = Path("./LOCATA_dcase/results").resolve()

    # Extract features if not done already
    if not len(lab_path.rglob("*.npy")) > 0:
        dev_feat_cls = cls_feature_class.FeatureClass(params)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

        # Extract labels
        dev_feat_cls.extract_all_labels()

    # Create test dataset
    data_gen_test = DataGenerator(
        params=params, split=2, shuffle=False, per_file=True
    )

    # Define device as whatever is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize evaluation metric class
    score_obj = ComputeSELDResults(params)

    # Create model
    data_in, data_out = data_gen_test.get_data_sizes()
    model = SeldModel(data_in, data_out, params).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    criterion = MSELoss_ADPIT()

    # Dump results in DCASE output format for calculating final scores
    dcase_output_test_folder = os.path.join(params['dcase_output_dir'], out_name, strftime("%Y%m%d%H%M%S", gmtime()))
    cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
    print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

    test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, "cpu")

    use_jackknife = True
    test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )
    print('\nTest Loss')
    print('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
    print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0]  if use_jackknife else test_ER, '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0], test_ER[1][1]) if use_jackknife else '', 100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
    print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '', 100*test_LR[0]  if use_jackknife else 100*test_LR,'[{:0.2f}, {:0.2f}]'.format(100*test_LR[1][0], 100*test_LR[1][1]) if use_jackknife else ''))
    if params['average']=='macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                 cls_cnt,
                 classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                 classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                 classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                 classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                 classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
