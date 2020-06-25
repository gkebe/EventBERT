import argparse
import subprocess
def main(args):
    def create_record_worker(filename_prefix, shard_id, output_format='hdf5'):
        hdf5_tfrecord_folder_prefix = "_lower_case_" + str(args.do_lower_case) + "_seq_len_" + str(args.max_seq_length) \
                                      + "_max_pred_" + str(args.max_predictions_per_seq) + "_masked_lm_prob_" + str(args.masked_lm_prob) \
                                      + "_random_seed_" + str(args.random_seed) + "_dupe_factor_" + str(args.dupe_factor)
        bert_preprocessing_command = 'python ../create_pretraining_data.py'
        if args.input_dir:
            bert_preprocessing_command += ' --input_file=' + args.input_dir + filename_prefix + '_' + str(
                shard_id) + '.txt'
        else:
            bert_preprocessing_command += ' --input_file=' + args.dataset + "/step3/" + filename_prefix + '_' + str(shard_id) + '.txt'
        bert_preprocessing_command += ' --output_file=' + 'hdf5' + hdf5_tfrecord_folder_prefix + '/' + args.dataset + '/' + filename_prefix + '_' + str(shard_id) + '.' + output_format
        bert_preprocessing_command += ' --vocab_file=' + args.vocab_file
        bert_preprocessing_command += ' --do_lower_case' if args.do_lower_case else ''
        bert_preprocessing_command += ' --max_seq_length=' + str(args.max_seq_length)
        bert_preprocessing_command += ' --max_predictions_per_seq=' + str(args.max_predictions_per_seq)
        bert_preprocessing_command += ' --masked_lm_prob=' + str(args.masked_lm_prob)
        bert_preprocessing_command += ' --random_seed=' + str(args.random_seed)
        bert_preprocessing_command += ' --dupe_factor=' + str(args.dupe_factor)
        bert_preprocessing_process = subprocess.Popen(bert_preprocessing_command, shell=True)

        last_process = bert_preprocessing_process

        # This could be better optimized (fine if all take equal time)
        if shard_id % args.n_processes == 0 and shard_id > 0:
            bert_preprocessing_process.wait()
        return last_process

    output_file_prefix = args.dataset

    for i in range(args.n_training_shards):
        last_process = create_record_worker(output_file_prefix + '_training', i)

    last_process.wait()

    for i in range(args.n_test_shards):
        last_process = create_record_worker(output_file_prefix + '_test', i)

    last_process.wait()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for Everything BERT-related'
    )
    parser.add_argument("--dataset",
                        default="wiki_70k",
                        type=str,
                        required=False,
                        help="Specify a dataset!")
    parser.add_argument("--input_dir",
                        type=str,
                        required=False,
                        help="Specify a dataset!")
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
        '--masked_lm_prob',
        type=float,
        help='Specify the probability for masked lm',
        default=0.15
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
    args = parser.parse_args()
    main(args)