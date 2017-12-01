from model.ucf101.model_rladder import *
from analysis.quantitative import print_results
from analysis.qualitative import save_sequences
from analysis.layer_removal import test_layer_subsets
import tensorflow as tf
import cPickle


def train(resume_training=False):
    # If resuming previous training, load from disk
    if resume_training:
        MODEL.load(SAVE_PATH)

    # Train future frame predictor
    MODEL.train(
        x=TRAIN_PREPROCESSOR,
        batch_size=BATCH_SIZE,
        iterations=TRAIN_STEPS,
        save_path=SAVE_PATH,
        save_frequency=100,
    )


def test():
    # Load model from disk and perform testing
    MODEL.load(SAVE_PATH)
    errs_pred, errs_base = MODEL.test(
        x=TEST_PREPROCESSOR,
        batch_size=BATCH_SIZE,
        metric=('mse', 'psnr', 'dssim'),
    )

    # Save and print baseline/prediction errors
    cPickle.dump((errs_pred, errs_base), open(SAVE_ROOT + 'predictions.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    print_results('Baseline errors', errs_base)
    print_results('Prediction errors', errs_pred)


def run():
    # Load from disk and start predicting
    MODEL.load(SAVE_PATH)
    predictions = MODEL.run(
        x=TEST_PREPROCESSOR,
        batch_size=BATCH_SIZE,
        plot=True,
    )


def analyse():
    # Load results from disk
    errs_pred, errs_base = cPickle.load(open(SAVE_ROOT + 'predictions.pkl', 'rb'))

    # Print results
    print_results('Baseline errors', errs_base)
    print_results('Prediction errors', errs_pred)

    # Plot best predictions, perform layer removal
    MODEL.load(SAVE_PATH)
    indices, _, _ = save_sequences(MODEL, TEST_PREPROCESSOR, SAVE_ROOT + 'qual/')
    test_layer_subsets(MODEL, TEST_PREPROCESSOR, indices, SAVE_ROOT + 'lremoval/')


if __name__ == '__main__':
    with tf.device(DEVICE):
        train(resume_training=False)
        test()
        analyse()
        # run()
