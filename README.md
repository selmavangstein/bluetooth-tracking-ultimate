# bluetooth-tracking-ultimate
We develop a bluetooth system for tracking ultimate frisbee players
***write some more intro about the project here***
# Things that should probably be in this file, or at least documented somewhere (remove things when we write them in)
- Downloads, inputs/outputs, how to interpret results, how to make relevant changes
- beacon positions, datafile, ground truth, which tests to use in which order, some tuning, maybe recommended order
- should mention csv file structure, how it is updated
- (not sure what of this fits in here vs below)
- testing procedure, how to hook up wearables, how to get the data, how to feed it to the pipeline
- Where data is stored, what is raw data vs final results, which tests have been done, and which file corresponds to which test
- Acknowledge eric and aaron, and tell people to reach out to eric if they want to work on this
  
# How to use
  

# ?File structure? or project structure

# Pipeline overview
The post processing pipeline serves to move from raw testing datafiles containing distances from player to beacons, to a cleaned and refined dataset containing player coordinates over the duration of the test. 
The raw data generally has low noise levels and lies close to ground truth. It gets noisy whenever something is blocking the line-of-sight between player and beacon. The post processing aims to identify and handle these noisy areas to get accurate coordinate estimates.
We now aim to walk through the steps in the pipeline. In broad strokes, we start by processing the 1D distance measurements to get rid of noise in the data. We then use an alternative trilateration algorithm to determine player coordinates from the position measurements. Finally, we filter the coordinates.

## Distance Correction
We found a systematic overestimate of about 0.8 meters in our raw data, so we subtract this for our dataset. This might be different for different wearables, so in the future we recommend to test each sensor individually, and tune them individually.

## Outlier Removal @Bem - can you fill in the details here? Also feel free to rewrite for clarity
This algorithm aims to identify noisy areas and get rid of them. The algorithm is based on identifying deviations from a mean, and is biased towards deleting inflated measurements, beacuse we notice that the sensors usually overestimates the distance whenever an obstacle is blocking the signal.
We use a rolling window of 800 ms, and calculate a linear fit in this window. We then mark any datapoint that deviates more than 0.5 meters from this fitted line. In noisy windows the then recalculate the line of best fit by using the local minima. We find that in blocks of noise, the local minima still lies close to the ground truth, and the noise causes spikes above this line. We therefore go through a second time and uses this minimum fitted line to identify outliers, this time only flagging outliers above the fitted line. We finally replace the outliers with nan, unless the window is filled with noise - then we replace them with a point on the fitted line.

## Velocity Clamping
Frisbee players on a field can only move with a max velocity of about 11m/s. We use this to identify remaining outliers in the data. We loop over the datapoints, and calculate the difference in time and distance from the last valid neighbor. We use this to calculate the average velocity between the datapoints, and if this is too high, we flag the point as invalid. We keep doing this until no more invalid points are found. We then go through and replace the remaining points with nan.

We will now be left with some small 'floating' clusters of points surrounded by nan. When an obstacle is blocking the signal, we might get several consequtive inflated measurements, and these will not be caught by the velocity clamping. Because these smaller floating groups is probably also noise, we go through and delete them.

We recommend tuning the threshold for what a "small" group of points is, depending on the hardware in use, and where in the pipeline this step is applied. If it is applied on raw data, high variance in noisy areas mean that many of these small groups are real values, and we want to be careful with deleting them. However, the outlier removal changes the characteristics of the dataset, and leaves wider bumps rather than narrow spikes in noisy areas, and we can therefore be a lot more agressive with a higher threshold.

There is a question about whether it is best to simply delete, or try to replace the invalid points. Since we have data from four beacons, and only three are theoretically required to find a coordinate from distance measurements, we decided to delete the points. This way we are not feeding faulty measurements to the trilateration algorithm.  

Lastly, note that there is a strong forward bias in this algorithm, meaning that it favours early datapoints over later ones. If we encounter a wide step representing noise, we might not be able to cut this whole step out. When the data then steps back down to the undisturbed measurements, it will mark the first clean values as invalid, because they have moved to fast away from the noisy step. In the future, it might be possible to also run the algorithm in the other direction, and use some combination of the marked datapoints to figure out which ones are truly invalid.

## Smoothing
The last thing we do to our distance measurements is to smooth them out. We use an EMA (exponential moving average) algorithm with a window of 15 datapoints. This window can be tuned in the future, but be cautious about making this window too large, since this will cause the data to lack behind groundt truth. The players might be moving eratically, so the smoothing should not be very agressive.

## Trilateration

## Kalman Filtering

### Dynamic filter adjustment

### Velocity and acceleration clamping

### RTS Smoothing



# Recommended future work
This might be mostly covered above. Feel free to add stuff here!
