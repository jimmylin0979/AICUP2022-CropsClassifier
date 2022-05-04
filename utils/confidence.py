#
import torch
import torch.nn.functional as F
import numpy as np

#
from config import DefualtConfig


def get_confidence_score(model, loader, topN=5, use_gpu_index='-1', batch_size=32, outpu_file_path='./prediction-Confidence.csv'):
    '''
    '''

    # Set the model state to 'eval', we should not update any parameter here
    model.eval()

    device = torch.device(f'cuda:{use_gpu_index}' if torch.cuda.is_available(
    ) else'cpu') if use_gpu_index != -1 else torch.device('cpu')

    #
    with torch.no_grad():

        with open(outpu_file_path, 'w') as file:

            file.write(
                f'validset_index, topN, Ground_truth, 1, prob1, 2, prob2, 3, prob3, 4, prob4, 5, prob5\n')

            for batch_idx, batch in enumerate(loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = model(imgs.to(device))

                y_label = logits.argmax(dim=-1)

                # List of
                y_probs = [F.softmax(el, dim=0)
                           for i, el in zip(y_label, logits)]

                # record the prediction & ground truth for later review
                for i, _ in enumerate(y_label):

                    img_idx = (batch_idx) * batch_size + i
                    # img = ds_valid.__getitem__((batch_idx- 1) * config['batch_size'] + i)
                    # # img = np.array(img)

                    topN = 5
                    # topN_labels = y_probs[i].argsort()[-topN:].tolist()[::-1]
                    topN_values, topN_labels = y_probs[i].topk(topN)

                    #
                    file.write(f'{img_idx}, {topN}, {labels[i].item()}')
                    for i in range(topN):
                        file.write(
                            f', {topN_labels[i].item()}, {topN_values[i].item()}')
                    else:
                        file.write("\n")
