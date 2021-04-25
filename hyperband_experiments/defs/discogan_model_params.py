
"function (and parameter space) definitions for hyperband"
"binary classification with gradient boosting"

#from common_defs import *
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from pprint import pprint

# a dict with x_train, y_train, x_test, y_test
#from load_data import data

#from sklearn.ensemble import GradientBoostingClassifier as GB

def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v

    return new_params

#

trees_per_iteration = 5

# subsample is good for showing out-of-bag errors 
# when fitting in verbose mode, and probably not much else
space = {
    # 'learning_rate': hp.uniform('lr', 0.0001, 0.001),
    # 'num_layers': hp.quniform('ns', 2, 6, 1),
    # 'batch_size': hp.quniform('bs', 1, 64, 1),
    'width_factor': hp.uniform('width_factor', 0.5, 4.0),
}

def get_params():
    params = sample(space)
    return handle_integers(params)


def try_params(n_iterations, params):
    n_estimators = int(round(n_iterations * trees_per_iteration))
    print "n_estimators:", n_estimators
    pprint(params)

    from common_params import TASK_NAME, ARCH

    if ARCH == 'discogan':
        from discogan_with_risk_model import Disco_with_riskGAN as GeneralGANBound
    else:
        ARCH = 'distancegan'
        from distancegan_with_risk_model import Distance_with_riskGAN as GeneralGANBound

    model = GeneralGANBound()
    path_str = 'width_factor_' + str(params['width_factor'])
    # path_str = 'lr_' + str(params['learning_rate']) + '_num_layers_' + str(params['num_layers']) + '_batch_size_' + str(params['batch_size'])
    model.args.width_factor = params['width_factor']
    # model.args.learning_rate = params['learning_rate']
    # model.args.num_layers = params['num_layers']
    # model.args.batch_size = params['batch_size']
    model.args.task_name = TASK_NAME
    model.args.model_path = './models_27_9_5_' + model.args.task_name + '/' + path_str
    model.args.result_path = './results_27_9_5_' + model.args.task_name + '/' + path_str
    model.args.model_arch = ARCH

    if ARCH == 'distancegan':
        model.args.batch_size = 8

    if TASK_NAME == 'cityscapes':
        model.args.default_correlation_rate = 3

    if TASK_NAME == 'facades':
        model.args.default_correlation_rate = 7

    if TASK_NAME == 'maps':
        model.args.default_correlation_rate = 0.2

    if model.args.task_name.startswith('edges'):
        model.args.default_correlation_rate = 0.5

    if model.args.task_name.startswith('edges'):
        model.args.epoch_size = n_estimators
        full_loss = model.run()
        loss = full_loss[0][1].cpu().data.numpy()[0]
    else:
        model.args.epoch_size = n_estimators*10
        if TASK_NAME == 'facades':
            if ARCH == 'discogan':
                model.args.epoch_size *= 20
            else:
                model.args.epoch_size *= 5
        if TASK_NAME == 'maps':
            model.args.epoch_size *= 10
        full_loss = model.run()
        loss = full_loss[0][0].cpu().data.numpy()[0]

    print (full_loss[0][0].cpu().data.numpy()[0], full_loss[0][1].cpu().data.numpy()[0]), \
          (full_loss[1][0].cpu().data.numpy()[0], full_loss[1][1].cpu().data.numpy()[0]),

    #clf = GB(n_estimators=n_estimators, verbose=0, **params)
    return { 'loss': loss}
