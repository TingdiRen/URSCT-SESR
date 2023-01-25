import os
from dataset.data_loader_detail import *

def get_training_data(dir, img_options):
    assert os.path.exists(dir)
    return DataLoaderTrain(dir, img_options)

def get_training_SR_data(dir, img_options, scale):
    assert os.path.exists(dir)
    return DataLoaderTrainSR(dir, img_options, scale)

def get_validation_data(dir, img_options):
    assert os.path.exists(dir)
    return DataLoaderVal(dir, img_options)

def get_validation_SR_data(dir, img_options, scale):
    assert os.path.exists(dir)
    return DataLoaderValSR(dir, img_options, scale)

def get_test_data(dir, img_options):
    assert os.path.exists(dir)
    return DataLoaderTest(dir, img_options)

def get_test_SR_data(dir, img_options, scale):
    assert os.path.exists(dir)
    return DataLoaderTestSR(dir, img_options, scale)

def get_infer_data(dir, img_options):
    assert os.path.exists(dir)
    return DataLoaderInf(dir, img_options)

def get_infer_SR_data(dir, img_options, scale):
    assert os.path.exists(dir)
    return DataLoaderInfSR(dir, img_options, scale)
