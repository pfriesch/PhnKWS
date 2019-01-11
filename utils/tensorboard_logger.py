from tensorboardX import SummaryWriter

class WriterTensorboardX(SummaryWriter):
    def __init__(self, log_dir=None, comment='', **kwargs):
        super().__init__(log_dir, comment, **kwargs)
        self.step = 0
        self.mode = ''

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_scalar('{}/{}'.format(self.mode, tag), scalar_value, self.step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_scalars('{}/{}'.format(self.mode, main_tag), tag_scalar_dict, self.step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_histogram('{}/{}'.format(self.mode, tag), values, self.step, bins, walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_image('{}/{}'.format(self.mode, tag), img_tensor, self.step, walltime, dataformats)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_images('{}/{}'.format(self.mode, tag), img_tensor, self.step, walltime, dataformats)

    def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None, walltime=None, dataformats='CHW',
                             **kwargs):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_image_with_boxes('{}/{}'.format(self.mode, tag), img_tensor, box_tensor, self.step, walltime, dataformats, **kwargs)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_figure('{}/{}'.format(self.mode, tag), figure, self.step, close, walltime)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_video('{}/{}'.format(self.mode, tag), vid_tensor, self.step, fps, walltime)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_audio('{}/{}'.format(self.mode, tag), snd_tensor, self.step, sample_rate, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_text('{}/{}'.format(self.mode, tag), text_string, self.step, walltime)

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_pr_curve('{}/{}'.format(self.mode, tag), labels, predictions, self.step, num_thresholds, weights, walltime)

    def add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts,
                         false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None,
                         walltime=None):
        assert global_step is None, "use the actual writer object if you want to specify the global step"
        super().add_pr_curve_raw('{}/{}'.format(self.mode, tag), true_positive_counts, false_positive_counts, true_negative_counts,
                                 false_negative_counts, precision, recall, self.step, num_thresholds, weights,
                                 walltime)

    def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'):
        super().add_custom_scalars_multilinechart(tags, category, title)

    def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'):
        super().add_custom_scalars_marginchart(tags, category, title)