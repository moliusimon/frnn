import os
import scipy.misc as sm
import numpy as np


def save_sequences(model, preprocessor, path, indices=None):
    # Randomly select indices if not specified
    indices = np.random.randint(low=0, high=preprocessor.loader.num_samples, size=(50,)) if indices is None else indices

    # Get preprocessor subsample with the sequences of interest
    preprocessor.set_loader(
        preprocessor.loader.instantiate(sample_indices=indices)
    )

    # Get ground truth and predicted sequences
    gt = np.transpose(preprocessor.retrieve()[0], axes=[1, 0, 2, 3, 4]) / 2 + 0.5
    predictions = model.run(x=preprocessor, batch_size=10) / 2 + 0.5

    # Remove channels dimension for greyscale images
    if gt.shape[-1] == 1:
        predictions, gt = predictions[..., 0], gt[..., 0]

    # Save predictions
    for i, (g, p) in enumerate(zip(gt, predictions)):
        # Create sequence path if it does not exist
        s_path = path + 's' + str(i) + '/'
        if not os.path.exists(s_path):
            os.makedirs(s_path)
        print s_path + ' -> ' + str(indices[i])

        for j in range(10):
            sm.imsave(s_path + 'g' + str(j) + '.png', g[j])
            sm.imsave(s_path + 'g' + str(j+10) + '.png', g[10+j])
            sm.imsave(s_path + 'p' + str(j+10) + '.png', p[j])

    # Return indices and sequences
    return indices, predictions, gt