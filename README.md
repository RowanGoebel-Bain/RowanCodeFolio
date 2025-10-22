In this project I use data produced by Skycameras of  various Maine cities to train this model on whether auroa borealis is visible in the sky. It is programmed to recieve a live feed from the cameras local to Bethel and deliver an email 
when Auroras are visible. There have been subsequent iterations of model refining with new training data created by flagging and retraining on its mistakes.


Heat Map of where data points come from in the sky.
<img width="571" height="453" alt="image" src="https://github.com/user-attachments/assets/f82e552d-1c23-4a69-86aa-da1c7663a672" />
These data display the origin of the starts in the dataset.

With these stars we compared their metal content (Fe/H) to several alpha elements that are formed by fusion.
<img width="703" height="545" alt="image" src="https://github.com/user-attachments/assets/1e48ce8d-b75c-4b6c-ab2e-ef915842c11b" />
by comparing anomalies of each element- such as 7, the gray outliers of Carbon poor stars in this figure- I created a figure to describe which anomalies of alpha elements coorelate with eachother.

<img width="870" height="545" alt="image" src="https://github.com/user-attachments/assets/9dd7fdf2-9861-49a9-8ac6-d642a4033143" />
Here we see which pairs of alpha elements coorelate with eachother in their anomalies. With expected values being calculated by the probability of how many stars in the dataset fall into both anomalous groups, I compare this to the actual count of of data that fell into each category. We are left with a % change of what elements are anomalous together compared to the random distribution.
