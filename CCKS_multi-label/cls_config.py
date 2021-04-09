import argparse
import os

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo",default="",type=str,required=False)
    parser.add_argument("--pretrained_path",default="./pretrained_models/bert_base",type=str,required=False)
    parser.add_argument("--do_train",default=False,type=bool,required=False)
    parser.add_argument("--do_dev",default=False,type=bool,required=False)
    parser.add_argument("--do_test",default=True,type=bool,required=False)
    parser.add_argument('--num_labels',default=29,type=int,required=False)
    parser.add_argument("--resume_model_path",type=str,default="./out/cls_out/model",required=False)
    parser.add_argument("--resume_model",default=True,type=bool,required=False)
    parser.add_argument("--data_dir",default="./data",type=str,required=False)
    parser.add_argument("--train_name",default="event_entity_train_data_label.csv",type=str,required=False)
    parser.add_argument("--dev_name",default="dev.json",type=str,required=False)
    parser.add_argument("--dev_rate",default=0.2,type=float,required=False)
    parser.add_argument("--test_name",default="event_entity_dev_data.csv",type=str,required=False)
    parser.add_argument("--max_workers",default=15,type=int,required=False)
    parser.add_argument("--max_seq_len",default=512,type=int,required=False)
    parser.add_argument("--batch_size",default=16,type=int,required=False)
    parser.add_argument("--eval_batch_size",default=128,type=int,required=False)
    parser.add_argument("--epochs",default=1,type=int,required=False)
    parser.add_argument("--eval_steps",default=200,type=int,required=False)
    parser.add_argument("--out_dir",type=str,default="./out/cls_out")

    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=1e-3, type=float,
                        help="L2 regularization.")

    parser.add_argument("--seed",default=42,type=int,required=False)
    parser.add_argument("--do_lower_case",default=True,type=bool,required=False)
    config = parser.parse_args()
    return config
base_config = config()

if __name__=="__main__":
    print(os.path.isdir("./pretrained_models/bert_base"))
