import scipy.misc as sm
import os


def test_layer_subsets(model, preprocessor, indices, save_path):
    # Get samples to analyse
    preprocessor.set_loader(
        preprocessor.loader.instantiate(sample_indices=indices)
    )

    # Start layer removal process
    n_layers, preds = [14, 13, 11, 10, 8, 7, 5, 4], []
    for n in [14, 13, 11, 10, 8, 7, 5, 4]:
        preprocessor.loader.reset()
        model.topology[0].topology = model.topology[0].topology[:n]
        t_preds = model.run(x=preprocessor, batch_size=10) / 2 + 0.5
        preds.append(t_preds)

    # Remove color channel if images are greyscale
    preds = [p[..., 0] for p in preds] if preds[0].shape[-1] == 1 else preds

    # Save predictions
    for n, p in zip(n_layers, preds):
        for i, s in enumerate(p):
            # Create sequence path if it does not exist
            s_path = save_path + 's' + str(i) + '/'
            if not os.path.exists(s_path):
                os.makedirs(s_path)

            # Save sequence frames
            for j in range(10):
                sm.imsave(s_path + 'l' + str(n) + '_f' + str(j) + '.png', s[j])

