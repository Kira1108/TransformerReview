import tensorflow as tf


class PaperScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """This learning scheduler is implemented according to the original paper.
    This scheduler includes an d_model parameter and a warmup_steps parameter.

    When increasing d_model, the learning rate will be decreased.
    Before warmup steps is reached, the learning rate will be increased.
    After warmup steps, the learning rate will be decreased.
    """

    def __init__(self, d_model, warmup_steps, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):

        arg1 = tf.math.rsqrt(step)

        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def test_paper_scheduler():

    import matplotlib.pyplot as plt

    scheduler = PaperScheduler(d_model=512, warmup_steps=4000)

    learning_rates = scheduler(tf.range(40000, dtype=tf.float32))

    plt.plot(learning_rates)
    plt.show()

if __name__ == "__main__":
    test_paper_scheduler()
