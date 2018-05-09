import includes.clustering as clust
import includes.iohelper as iohelper
import includes.dataset_helper as dsf
import includes.feature_learning_constr as fl_constr

args = iohelper.parse_args()
rconf = iohelper.load_config(args)
# generate timestamp as a dir for run and directory (run_dir) to store readouts from that run
run_tag, run_dir = iohelper.setup_run(args, tag_name=rconf['tag_name'])
out_filename = dsf.replace_dir(rconf['test_file_name'], run_dir)
# Augument config
iohelper.process_config(run_tag, run_dir, args, rconf)

###############

if not args.load:
    print("training {} fold...".format(rconf['run_args']['k_fold']))
    feature_learn = fl_constr.create(rconf['type'], rconf['run_args'])
    feature_learn.train()
    feature_learn.load_best_check_point()
    feature_learn.extract_vectors_from_dataset(out_filename=out_filename)
    if feature_learn.dataset.params['srcNumRows'] > 0 and feature_learn.dataset.params['srcNumCols'] > 0:
        feature_learn.extract_vectors_from_dataset(out_filename=out_filename, type="all")
    if 'inference_file_name' in rconf:
        for testFileAug in rconf['inference_file_name']:
            feature_learn.dataset = dsf.read_data_sets(testFileAug, one_hot=False, reshape=False, shuffle=True,
                                                       validation_size=750)
            out_filename_aug = dsf.replace_dir(testFileAug, run_dir)
            feature_learn.extract_vectors_from_dataset(out_filename=out_filename_aug)
            if feature_learn.dataset.params['srcNumRows'] > 0 and feature_learn.dataset.params['srcNumCols'] > 0:
                feature_learn.extract_vectors_from_dataset(out_filename=out_filename, type="all")
    feature_learn.finish_up()
else:
    clust.cleanOldData(run_dir)

clust.clustering_main(out_filename, (run_tag, run_dir), dat_file=rconf['test_file_name'])
if 'inference_file_name' in rconf:
    for testFileAug in rconf['inference_file_name']:
        out_filename_aug = dsf.replace_dir(testFileAug, run_dir)
        clust.clustering_main(out_filename_aug, (run_tag, run_dir))
