from data.data_loader import CreateDataLoader
from models.models import create_model
from test import test
from options.test_options import TestOptions
from util.visualizer import Visualizer

if __name__ == '__main__':
    # TODO what is save?
    opt = TestOptions().parse(save=False)
    opt.display_id = 0  # do not launch visdom
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.in_the_wild = True  # This triggers preprocessing of in the wild images in the dataloader
    opt.traverse = True  # This tells the model to traverse the latent space between anchor classes
    opt.interp_step = 0.05  # this controls the number of images to interpolate between anchor classes

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)

    opt.name = 'males_model'  # change to 'females_model' if you're trying the code on a female image
    model = create_model(opt)
    model.eval()

