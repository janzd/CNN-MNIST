import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import mode

filenames = [file for file in os.listdir("../results") if file.endswith(".csv")]
predictions = [pd.read_csv("../results/" + filename) for filename in filenames]
prediction_values = [prediction.values for prediction in predictions]
label = list(predictions[0])[1:]
ids = list(predictions[0]['ImageId'].values)
prediction_values = np.array(prediction_values)[:,:,1:]
average_prediction = mode(prediction_values)[0][0]
average_prediction = pd.DataFrame(average_prediction, index=ids, columns=label)
average_prediction.index.name = 'ImageId'

fp = open('../results/mnist_predictions_ensembleof%d_%s.csv' % (len(predictions), datetime.now().strftime('%Y-%m-%d_%H%M')),'w')
fp.write(average_prediction.to_csv())
fp.close 