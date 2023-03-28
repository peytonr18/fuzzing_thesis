import logging
import sys
import json
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score, roc_curve
import matplotlib.pyplot as plt

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=float(label.strip('[]'))
    return predictions


def read_binary_predictions(filename):
    binary_predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            binary_predictions[int(idx)]=int(label)
    return binary_predictions


def calculate_scores(answers,binary_predictions):
    Acc=[]
    for key in answers:
        if key not in binary_predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key]==binary_predictions[key])

    scores={}
    scores['Acc']=np.mean(Acc)
    return scores

#def calculate_PRAUC(answers, predictions):
  #  y_true=[]
  #  y_scores=[]


 #   answers_fit=list(answers.values())
 #   y_true = answers_fit
    
 #   predict_fit=list(predictions.values())
 #   y_scores=predict_fit

       # y_true.append(answers[key]])
       # y_true.append[predictions[key]])
#    return y_true


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions_binary', '-b',help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    binary_predictions=read_binary_predictions(args.predictions_binary)

    scores=calculate_scores(answers,binary_predictions)
   # prauc=calculate_PRAUC(answers,predictions)

    answers_fit=list(answers.values())
    predict_fit=list(predictions.values())

   # print(answers_fit)
    #print(predict_fit)
    print(scores)
    #print(prauc)
    
 # PR-AUC Curve
    y_true=[]
    y_scores=[]
    y_true = answers_fit
    y_scores=predict_fit
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
   
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'PR-AUC={pr_auc:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')

    fig.savefig('pr_auc_curve.png')
    plt.clf()

 # F1 SCORE
    predict_binary_fit=list(binary_predictions.values())
    y_binary_scores = predict_binary_fit
    f1 = f1_score(y_true, y_binary_scores)
    print(f"F1-score: {f1:.2f}")

 # Confusion Matrix
    cm = confusion_matrix(y_true, y_binary_scores)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], '.2f'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.savefig('confusion_matrix.png')
    plt.clf()
 # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    plt.savefig('roc_curve.png')
    plt.clf()


if __name__ == '__main__':
    predictions={}
    main()
