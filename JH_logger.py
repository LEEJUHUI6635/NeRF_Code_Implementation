from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    def scalar_writer(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    def image_writer(self, tag, image):
        img_grid = make_grid(image, normalize=True)
        self.writer.add_image(tag, img_grid)