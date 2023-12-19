# Vibration Analysis of Dams using a Convolutional Neural Network (CNN)

* Receives spectrograms as input.
* Outputs the first five natural frequency values.

## Quick start

Python 3.10

```
pip install -r requirements.txt
```

Extract all zips inside the data folder into a single folder.

```
$ python train.py
$ python plots.py
```

## Best trained model

* Fed with approximately 20,000 hourly spectrograms of size 801 × 16 × 3 (801 amplitude values collected per hour by 16 sensors).
* Lowest validation error: 7.44 × 10<sup>-5</sup> after 10,000 epochs.
