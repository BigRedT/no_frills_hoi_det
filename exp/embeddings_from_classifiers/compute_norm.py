import torch
from exp.embeddings_from_classifiers.load_classifiers import \
    load_pretrained_hoi_classifier


def main():
    print('Loading Classifier ...')
    classifiers,_ = load_pretrained_hoi_classifier()
    print('Mean squared classifier weights')
    for factor,classifier in classifiers.items():
        mean_squared_value = torch.mean(classifier*classifier)
        print(f'{factor}: {mean_squared_value.data[0]}')

        
if __name__=='__main__':
    main()
