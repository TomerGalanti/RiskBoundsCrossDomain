import time
import os
from datetime import datetime
from tensorboardX import SummaryWriter
from cyclegan_arch.cyclegan_arch_options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from cyclegan_arch.data.data_loader import CreateDataLoader
from cyclegan_arch.models import create_model
from cyclegan_arch.util.visualizer import Visualizer
from cyclegan_arch.data.aligned_data_loader import AlignedDataLoader

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

data_loader_2 = AlignedDataLoader()
data_loader_2.initialize(opt, test=True)
dataset_2 = iter(data_loader_2.load_data())

model = create_model(opt, dataset=dataset)
visualizer = Visualizer(opt)

total_steps = 0

# tensorboard_logger.configure("logs/log_" + time.ctime())
the_time = datetime.now().strftime('%B%d  %H:%M:%S')
writer_1 = SummaryWriter(os.path.join("logs/", opt.checkpoints_dir, opt.name, 'logs/' + the_time + '/1.log'))
writer_2 = SummaryWriter(os.path.join("logs/", opt.checkpoints_dir, opt.name, 'logs/' + the_time + '/2.log'))

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)

        # If data doesn't devide batch size this causes batches of size 1
        # which are problamatic for distance comparison
        if len(data['A']) != opt.batchSize or len(data['B']) != opt.batchSize:
            continue

        try:
            data_2 = dataset_2.next()
        except StopIteration:
            data_loader_2.initialize(opt, test=True)
            dataset_2 = iter(data_loader_2.load_data())
            data_2 = dataset_2.next()

        model.set_input(data, data_2)
        # model.set_input(data, data)
        model.optimize_parameters(iter=i)

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors(total_steps=total_steps, writer_1=writer_1, writer_2=writer_2)
            t = (time.time() - iter_start_time) / opt.batchSize
            # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(total_steps)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
