#! /bin/bash

for fold in 0 1 2 3 4 5 6 7 8 9
do
    python trainClassifier.py --fold $fold
    for noveltyDataset in "None"
    do
        python test_softmax.py --NoveltyDataset  $noveltyDataset --fold $fold
        python test_mcd.py --NoveltyDataset  $noveltyDataset --fold $fold
        python test_odin.py --NoveltyDataset  $noveltyDataset --fold $fold
        python test_maxLogit.py --NoveltyDataset  $noveltyDataset --fold $fold
        python test_energy.py --NoveltyDataset  $noveltyDataset --fold $fold
        python test_entropy.py --NoveltyDataset  $noveltyDataset --fold $fold
    done
done
