import argparse
import subprocess
def main(args):
    def create_record_worker(filename_prefix, shard_id, output_format='hdf5'):
        hdf5_tfrecord_folder_prefix = "_lower_case_" + str(args.do_lower_case) + "_seq_len_" + str(args.max_seq_length) \
                                      + "_max_pred_" + str(args.max_predictions_per_seq) \
                                      + "_random_seed_" + str(args.random_seed) + "_dupe_factor_" + str(args.dupe_factor)
        xlnet_preprocessing_command = 'python ../create_pretraining_data_xlnet.py'
        xlnet_preprocessing_command += ' --input_file=' + "wiki_70k/step3/" + filename_prefix + '_' + str(shard_id) + '.txt'
        xlnet_preprocessing_command += ' --output_file=' + 'hdf5' + '_xlnet_' + hdf5_tfrecord_folder_prefix + '/' + "wiki_70k" + '/' + filename_prefix + '_' + str(shard_id) + '.' + output_format
        xlnet_preprocessing_command += ' --vocab_file=' + args.vocab_file
        xlnet_preprocessing_command += ' --do_lower_case' if args.do_lower_case else ''
        xlnet_preprocessing_command += ' --max_seq_length=' + str(args.max_seq_length)
        xlnet_preprocessing_command += ' --max_predictions_per_seq=' + str(args.max_predictions_per_seq)
        xlnet_preprocessing_command += ' --random_seed=' + str(args.random_seed)
        xlnet_preprocessing_command += ' --dupe_factor=' + str(args.dupe_factor)
        xlnet_preprocessing_process = subprocess.Popen(xlnet_preprocessing_command, shell=True)

        last_process = xlnet_preprocessing_process

        # This could be better optimized (fine if all take equal time)
        if shard_id % args.n_processes == 0 and shard_id > 0:
            xlnet_preprocessing_process.wait()
        return last_process

    output_file_prefix = "wiki_70k"

    for i in range(args.n_training_shards):
        last_process = create_record_worker(output_file_prefix + '_training', i)

    last_process.wait()

    for i in range(args.n_test_shards):
        if args.keep_label:
            last_process = create_record_worker(output_file_prefix + '_test_with_labels', i)
        else:
            last_process = create_record_worker(output_file_prefix + '_test', i)
    last_process.wait()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for Everything XLNet-related'
    )

    parser.add_argument(
        '--n_training_shards',
        type=int,
        help='Specify the number of training shards to generate',
        default=210
    )

    parser.add_argument(
        '--n_test_shards',
        type=int,
        help='Specify the number of test shards to generate',
        default=1
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        help='Specify the base seed to use for any random number generation',
        default=12345
    )

    parser.add_argument(
        '--dupe_factor',
        type=int,
        help='Specify the duplication factor',
        default=5
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        help='Specify the maximum sequence length',
        default=512
    )

    parser.add_argument(
        '--max_predictions_per_seq',
        type=int,
        help='Specify the maximum number of masked words per sequence',
        default=20
    )

    parser.add_argument(
        '--do_lower_case',
        type=int,
        help='Specify whether it is cased (0) or uncased (1) (any number greater than 0 will be treated as uncased)',
        default=1
    )

    parser.add_argument(
        '--vocab_file',
        type=str,
        help='Specify absolute path to vocab file to use)'
    )

    parser.add_argument(
        '--skip_wikiextractor',
        type=int,
        help='Specify whether to skip wikiextractor step 0=False, 1=True',
        default=0
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        help='Specify the max number of processes to allow at one time',
        default=4
    )
    parser.add_argument(
        "--keep_label",
        default=False,
        type=bool,
        required=False,
        help="Specify a output filename!"
    )
    args = parser.parse_args()
    main(args)