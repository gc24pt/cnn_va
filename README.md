# CNN for Vibration Analysis

* Receives spectrograms as input.
* Outputs the first five natural frequency values.

## Best trained model (model.rar)

* Fed with approximately 20,000 hourly spectrograms of size 801 × 16 × 3 (801 amplitude values collected per hour by 16 sensors).
* Lowest validation error: 7.44 × 10<sup>-5</sup> after 10,000 epochs.
