
def create_model(opt, dataset=None):

    print(opt.model)

    if opt.model == 'gan':
        from .gan_model import GANModel
        model = GANModel()
    elif opt.model == 'cycle_gan_with_risk':
        from .cycle_gan_model import CycleGANModelWithRisk
        model = CycleGANModelWithRisk()
    elif opt.model == "distance_gan_with_risk":
        from .distance_gan_model import DistanceGANModelWithRisk
        model = DistanceGANModelWithRisk(dataset)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
