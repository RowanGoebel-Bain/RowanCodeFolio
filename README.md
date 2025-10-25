##ML Binary Classification Aurora Borealis Detection
In this project I use data produced by Skycameras of  various Maine cities to train this model on whether auroa borealis is visible in the sky using tensorflow. It is programmed to recieve a live feed from the cameras local to Bethel and deliver an alert when Auroras are visible. There have been subsequent iterations of model refining with new training data created by flagging and retraining on its mistakes. 99% accuracy was obtained by the recent model.


##Clustering Stars by Metallicity
In this project we draw stellar data LAMOST's data release 9, and inspect the stellar chemistry to anomalous behavior with respect to elemental composition. The code for comparing different alpha elements (C, Ca, N, Ni Mg< Al) is located inclustering2 and anomalous correlations appear in artificial Clusters&result figs.
<img src="https://github.com/user-attachments/assets/f82e552d-1c23-4a69-86aa-da1c7663a672" alt="image" width="571" height="453" style="float: right; margin-left: 20px; margin-bottom: 20px;">
Heat Map of where data points come from in the sky. These data display the origin of the starts in the dataset.

<img src="https://github.com/user-attachments/assets/1e48ce8d-b75c-4b6c-ab2e-ef915842c11b" alt="image" width="703" height="545" style="float: right; margin-left: 20px; margin-bottom: 20px;">
With these stars we compared their metal content (Fe/H) to several alpha elements that are formed by fusion. By comparing anomalies of each element- such as 7, the gray outliers of Carbon poor stars in this figure- I created a figure to describe which anomalies of alpha elements coorelate with eachother which is shown below. Preceding the final figure, there are 5 other charts not included here that provide outliers for each of the alpha elements

<img src="https://github.com/user-attachments/assets/9dd7fdf2-9861-49a9-8ac6-d642a4033143" alt="image" width="870" height="545" style="float: right; margin-left: 20px; margin-bottom: 20px;">
Here we see which pairs of alpha elements coorelate with eachother in their anomalies. With expected values being calculated by the probability of how many stars in the dataset fall into both anomalous groups, I compare this to the actual count of of data that fell into each category. We are left with a % change of what elements are anomalous together compared to the random distribution.

<img src="https://github.com/user-attachments/assets/bbb419ce-0b81-450f-b4bd-d8368a8d397e" alt="image" width="1072" height="495" style="float: right; margin-left: 20px; margin-bottom: 20px;">
expiramented with three dimensional visualizations but didnt end up doingwork heavily with it in this project
