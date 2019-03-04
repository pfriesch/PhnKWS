from tensorboardX import SummaryWriter, FileWriter


class WriterTensorboardX(SummaryWriter):
    def __init__(self, log_dir=None, comment='', **kwargs):
        super().__init__(log_dir, comment, **kwargs)
        self.step = 0
        self.mode = ''
        self.kwargs = kwargs
        self.kwargs['comment'] = comment

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_scalar(f'{mode}/{tag}', scalar_value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_scalars(f'{mode}/{main_tag}', tag_scalar_dict, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_histogram(f'{mode}/{tag}', values, global_step, bins, walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW', mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_image(f'{mode}/{tag}', img_tensor, global_step, walltime, dataformats)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW', mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_images(f'{mode}/{tag}', img_tensor, global_step, walltime, dataformats)

    def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None, walltime=None, dataformats='CHW',
                             mode=None,
                             **kwargs):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_image_with_boxes(f'{mode}/{tag}', img_tensor, box_tensor, global_step, walltime,
                                     dataformats, **kwargs)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_figure(f'{mode}/{tag}', figure, global_step, close, walltime)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_video(f'{mode}/{tag}', vid_tensor, global_step, fps, walltime)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_audio(f'{mode}/{tag}', snd_tensor, global_step, sample_rate, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_text(f'{mode}/{tag}', text_string, global_step, walltime)

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None,
                     mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_pr_curve(f'{mode}/{tag}', labels, predictions, global_step, num_thresholds, weights,
                             walltime)

    def add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts,
                         false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None,
                         walltime=None, mode=None):
        if global_step is None:
            global_step = self.step
        if mode is None:
            mode = self.mode
        super().add_pr_curve_raw(f'{mode}/{tag}', true_positive_counts, false_positive_counts,
                                 true_negative_counts,
                                 false_negative_counts, precision, recall, global_step, num_thresholds, weights,
                                 walltime)

    def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'):
        super().add_custom_scalars_multilinechart(tags, category, title)

    def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'):
        super().add_custom_scalars_marginchart(tags, category, title)
