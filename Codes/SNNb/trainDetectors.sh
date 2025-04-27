for i in {0..9}
do
    for j in {0..2}
    do
        python -u trainDetector.py --detectorClass $j --fold $i
    done
    python -u ComputeMetrics.py --fold $i
done
