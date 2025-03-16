# Evaluating Methods for Radio Frequency Based Positioning of Ultimate Frisbee Players

## Abstract
We developed a bluetooth system for tracking ultimate frisbee players. Our project was centered around:
- Exploring different technologies
- Creating and optimizing physical implementations of the best technologies (FTM and UWB), complete with beacons, mounts, and wearables
- Process and analyze the data to make it cleaner and more accurate

This repo contains active, final implementations of data processing and analysis including the data pipeline and the ground truth system, in addition to Arduino and CAD files for the physical implementations. 

### Contributors

| Name           | Email                 |
| -------------- | --------------------- |
| Cullen Baker       | bakerc@gmail.com   |
| Will Shrestha | shresthaw@carleton.edu |
| Selma Vangstein     | vangsteins@carleton.edu |
| Marshall Johnson  | johnsonm@carleton.edu   |
| Bemnet Abebayehu  | abebayehub@carleton.edu |

## Contents

- [Abstract](#abstract)
- [How to use](#how-to-use)
- [Project Structure](#project-structure)
- [Data Processing overview](#data-processing-overview)
- [Ground Truth Overview](#ground-truth-overview)
- [Recommended Future Work](#recommended-future-work)
- [Acknowledgements](#acknowledgements)

## Things that should probably be in this file, or at least documented somewhere (remove things when we write them in)
- Downloads, inputs/outputs, how to interpret results, how to make relevant changes
- beacon positions, datafile, ground truth, which tests to use in which order, some tuning, maybe recommended order
- should mention csv file structure, how it is updated
- (not sure what of this fits in here vs below)
- testing procedure, how to hook up wearables, how to get the data, how to feed it to the pipeline
- Where data is stored, what is raw data vs final results, which tests have been done, and which file corresponds to which test
- Acknowledge eric and aaron, and tell people to reach out to eric if they want to work on this
  
## How to use
Simply run `final-postprocessing-pipeline.py` on your desired output by editing the file names within main, and it will apply the processing steps one piece at a time. 

## Project Structure

To navigate through our codebase, follow the steps below:

1. **Data Input:**
- We start with raw data files (in `.log` format). These files contain distance measurements from players to beacons.
- Example files: `rawdataexample.log` (raw data) and `rawdataexample.csv` (processed data).
- All data is stored under the `data/` folder.

2. **Data Cleaning:**
- First, we clean and convert the raw data to a CSV format for easier processing. The folder `data/` contains both ground truth data and processed data files for various days tests were conducted.
- Once our data is retrieved and in our desired format, we process the data through various cleaning algorithms in this order:
    - **Distance Correction**
    - **Outlier Removal**
    - **Velocity Clamping**
    - **Smoothing**
    - **Velocity Clamping**
- NOTE: More details for each algorithm above can be found undernearth [Data Processing overview](#data-processing-overview). 
3. **Post-Processing Pipeline:**
- The `final-postprocessing-pipeline.py` file contains scripts that sequentially apply each algorithm. Each step is executed one after the other, and users can rearrange the order of algorithms to fit their needs.
- After cleaning, the data is used to estimate player positions using trilateration, which can be found in `final-trilateration.py`.

4. **Kalman Filtering and Final Processing:**
- The cleaned coordinates are further refined using a Kalman filter. This is implemented in `Final-kalman.py`.
- NOTE: More details for the Kalman filter can be found underneath [Kalman filtering ](#kalman-filtering).


5. **Charts:**
- After the data has been processed, various charts are generated to visualize the effectiveness of each post-processing step. These charts are stored in the `charts/` folder.
- You will find:
    - Plots that show the behavior of the data before and after applying each algorithm.
    - An animation (MP4 video) under the name "streamlit" which visualizes the player’s path during the test.

6. **Frontend:**
- The `frontend.py` file contains a Streamlit-based web interface where you can visualize the results and generate PDF reports from the processed data.

## Data Processing overview
The data processing serves to move from raw testing data files containing distances from player to beacons, to a cleaned and refined dataset containing player coordinates over the duration of the test. 
The raw data generally has low noise levels and lies close to ground truth. It gets noisy whenever something is blocking the line-of-sight between player and beacon. The post processing aims to identify and handle these noisy areas to get accurate coordinate estimates. 
We now aim to walk through the steps in the pipeline. In broad strokes, we start by processing the 1D distance measurements to get rid of noise in the data. The steps in this process are applied in the order listed below, which is what we think is the best order, but the code enables a user to apply these in any order. We then use an alternative trilateration algorithm to determine player coordinates from the position measurements. Finally, we filter the coordinates.

### Distance Correction
We found a systematic overestimate of about 0.8 meters in our raw data, so we subtract this for our dataset. This might be different for different wearables, so in the future we recommend to test each sensor individually, and tune them individually.

### Outlier Removal
This algorithm aims to identify noisy areas and get rid of them. The algorithm is based on identifying deviations from a mean, and is biased towards deleting inflated/deflated measurements, because we notice that the sensors usually overestimates the distance whenever an obstacle is blocking the signal. 
We use a rolling window of 800 ms, and calculate a linear fit in this window. We then mark any window containing a datapoint that deviates more than 0.5 meter from this fitted line as a window containing an outlier. In windows flagged to contain an outlier, we then calculate the median of the window and take the difference of each data point against the median and if it deviates by more than half a meter in either +/- direction, targeting those large spikes, we exclude those points and created a fitted line with the remaining data points. We find that in blocks of noise, there are real data points that still lie close to the ground truth, and the noise causes spikes above this line. We therefore go through a second time in the same manner, calculating the median of the adjusted window, this time only flagging outliers above the fitted line that deviate by more than 0.5 meter in the + direction. We exclude those flagged points and return our fitted line with this newly adjusted window and replace the outliers with nan, unless the window is filled with noise - then we replace them with a point on the fitted line.

### Velocity Clamping
Frisbee players on a field can only move with a max velocity of about 11m/s. We use this to identify remaining outliers in the data. We loop over the data points, and calculate the difference in time and distance from the last valid neighbor. We use this to calculate the average velocity between the data points, and if this is too high, we flag the point as invalid. We keep doing this until no more invalid points are found. We then go through and replace the remaining points with nan.

We will now be left with some small 'floating' clusters of points surrounded by nan. When an obstacle is blocking the signal, we might get several consecutive inflated measurements, and these will not be caught by the velocity clamping. Because these smaller floating groups are probably also noise, we go through and delete them.

We recommend tuning the threshold for what a "small" group of points is, depending on the hardware in use, and where in the pipeline this step is applied. If it is applied on raw data, high variance in noisy areas means that many of these small groups are real values, and we want to be careful with deleting them. However, the outlier removal changes the characteristics of the dataset, and leaves wider bumps rather than narrow spikes in noisy areas, and we can therefore be a lot more aggressive with a higher threshold.

There is a question about whether it is best to simply delete, or try to replace the invalid points. Since we have data from four beacons, and only three are theoretically required to find a coordinate from distance measurements, we decided to delete the points. This way we are not feeding faulty measurements to the trilateration algorithm.

Lastly, note that there is a strong forward bias in this algorithm, meaning that it favours early data points over later ones. If we encounter a wide step representing noise, we might not be able to cut this whole step out. When the data then steps back down to the undisturbed measurements, it will mark the first clean values as invalid, because they have moved too fast away from the noisy step. In the future, it might be possible to also run the algorithm in the other direction, and use some combination of the marked data points to figure out which ones are truly invalid.

### Smoothing
The last thing we do to our distance measurements is to smooth them out. We use an EMA (exponential moving average) algorithm with a window of 15 data points. This window can be tuned in the future, but be cautious about making this window too large, since this will cause the data to lack behind ground truth. The players might be moving erratically, so the smoothing should not be very aggressive.

### Trilateration
The next step is to convert the distance measurements into player coordinates. Regular trilateration is very sensitive to erroneous measurements, so we developed our own algorithm. 
For each set of four distance measurements, we do the following:
- Draw a circle around each beacon corresponding to the measured beacon
- Mark all circle intersections
- Identify the densest cluster of circle intersections
- Find the centroid of this densest cluster. This is the final position coordinate.
- Use the distance between the coordinate and each circle to calculate a measurement error
This algorithm is more robust than regular trilateration - if one beacon measurement is off, it will not affect the final coordinate. We will also still be able to come up with a coordinate if some data is completely missing. 
We do this for each timestamp, and finally plot a player trace with the coordinates.

### Kalman Filtering
We now apply a Kalman filter to the coordinate dataset. We use a 6-state constant acceleration model, with (x,y)-position coordinates as the only measurement, and velocity and acceleration as hidden variables. 
A Kalman filter is essentially a cycle of predictions, measurements and weighted averages. The filter uses a constant acceleration model to use the previous measurement to predict what the next position will be. It then takes in the actual measured coordinates, and uses the uncertainties of the measurement and prediction to take a weighted average between the two, and this gives us the filtered coordinate. 
The repo we used, along with some really thorough documentation, can be found here: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

We also added some additional steps to the filter cycle:

#### Dynamic filter adjustment
We used the error estimation from the trilateration algorithm to dynamically adjust the filter as we go. If the coordinate error is too high, we inflated the measurement uncertainty, essentially telling the filter to mostly disregard the measurement, and trust the prediction instead.

#### Velocity and acceleration clamping
We then added a 2D-version of the velocity clamping from above. This time, we did not want gaps in our data, so whenever an invalid point was detected, we replaced it with a new point in the same direction, but closer to its neighbor. We also added another layer to this clamping, taking the current acceleration magnitude into account. This magnitude was calculated from the data we get from the accelerometer.

#### RTS Smoothing
The final step is to apply RTS Smoothing. This smoothing algorithm is implemented for us by the Kalman package. It work in essentially the same way as the Kalman filter, except for the fact that it runs backwards. This means that it uses the future datapoints to predict the previous ones, balancing out the forward bias the Kalman filter introduced.

## Ground Truth Overview
We recorded videos of our testing. Each video begins by looking at a phone that is on time.gov displaying the exact time. This allows us to match where the testing subject is in the video to the time of the outputted data. 

We used a couple different systems of actually obtaining the ground truth data for each timestamp, all of which are centered around Google Sheets and the distance between cones and beacons. One method is to assign grid coordinates to the cones, then use the distance between two points formula for each (works best when the path walked varies). Another is to precalculate the distance between each cone and each beacon and then copy it for each additional reading (works best when the path walked stays the same)

Our ground truth system is fundamentally imperfect as, since we have humans inputting ground truth data that has been matched to the second, each ground truth reading can vary by up to one second.

The ground truth data is then compared to the raw data and the processed data using GroundTruthPipeline.py.

The videos, Google Sheets files, raw data, and outputted ground truth can all be found in this repository. 

## Recommended Future Work
This might be mostly covered above. Feel free to add stuff here!

## Acknowledgements
We would like to thank our advisor Eric Alexander for all of the patience and help given to us throughout these past two terms. He has been instrumental in guiding us in completing our project. We would also like to thank Aaron Heidgerken-Greene for his assistance and help within the makerspace and constructions of our hardware. Please feel free to reach out to Eric Alexander for any questions or wants to work on this. 