# encoding=utf-8
import argparse
from data_loaders import *

def load_data(config, mode="train"):
    if config.data_type=="ner":
        return load_data_ner(config, mode)
    elif config.data_type=="w2v":
        return load_data_w2v(config, mode)
    else:
        return load_data_clf(config, mode)

if __name__=="__main__":
    from tf_utils import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default=".")
    args = parser.parse_args()
    config=load_config(args.config_path)
    data=load_data(config)
    for dd in [data.train_data, data.dev_data]:
        print dd
        for k,inputs in enumerate(dd):
            print '-'*20,'batch ',k,'-'*20
            for inp in inputs:
                print inp.shape, inp.dtype
            # if k>=20:break
