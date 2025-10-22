In this project I use data produced by Skycameras of  various Maine cities to train this model on whether auroa borealis is visible in the sky. It is programmed to recieve a live feed from the cameras local to Bethel and deliver an email 
when Auroras are visible. There have been subsequent iterations of model refining with new training data created by flagging and retraining on its mistakes.

Point distribution for Fe/H CNN by alpha m lasp
<img width="816" height="545" alt="image" src="https://github.com/user-attachments/assets/f98c8e9f-c584-4b81-9841-40482b26eca7" />



Heat Map of where data points come from in the sky.
<img width="571" height="453" alt="image" src="https://github.com/user-attachments/assets/f82e552d-1c23-4a69-86aa-da1c7663a672" />


logg_lasp:       From LASP, measures how compact the star is (higher logg for dwarfs, lower for giants).
gaia_g_mean_mag: Mean magnitude in the Gaia G band (broad visible bandpass, ~330-1050 nm) from Gaia DR9; represents the average brightness of the star in this filter.
rv_br1:          Radial velocity (km/s) from combined blue and red after zero-point correction.

feh_cnn:         Metallicity [Fe/H] from CNN.
alpha_m_lasp:    Alpha element abundance [α/M] from LASP; represents the average overabundance of alpha elements (like Mg, Si, Ca, Ti) relative to metallicity, often indicating stellar population origins
