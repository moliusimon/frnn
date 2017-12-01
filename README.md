Here we explain how the code is structured.


DATASETS & MODELS:
The models for each dataset are saved inside "./model/<dataset>/". Each model has a main file, one for the fRNN model
named "model_frnn.py" and one for the RLadder baseline, named "main_rladder.py". These files specify the paths where
to find the pre-processed data and save the trained models, as well as the training parameters:

- Number of training terations (batches)
- Batch size
- Device to use
- Topology parameters
- Data loading and augmentation parameters

The folder for each dataset also contains a dataset-specific loader to feed the network ("loader.py") and a data
preprocessing script ("preprocess.py"). The former should only be modified if you plan on using the code on other
datasets not considered here. The later should be manually run in order to prepare the dataset before trying
to train any model. In the case of Moving MMNIST, the preprocessing script will download the necessary files before
preprocessing. For KTH and UCF101 the script expects the uncompressed datasets to be already present.


MAIN FILES:
Each model has a main file associated. They are all identical, changing only the imported model file. By default, the
model will train, test and analyse the results of the model. Each step can be executed independently by commenting the
other actions, as intermediate results are saved to disk. There is also a "run" function commented by default. This
function allows you to extract direct predictions from the model, as well as to plot the results as they are obtained.