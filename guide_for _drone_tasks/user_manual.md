- [Section 1: Creation and export of the flight paths](#section-1-creation-and-export-of-the-flight-paths)
- [Section 2: Uploading the flight path to the drone or drone app](#section-2-uploading-the-flight-path-to-the-drone-or-drone-app)
  * [For DJI Mini 5 Pro](#for-dji-mini-5-pro)
- [Section 3: Flying the drone](#section-3-flying-the-drone)
  * [DJI Mini 3](#dji-mini-3)
- [Section 4: Creation of orthomosaic in WebODM](#section-4-creation-of-orthomosaic-in-webodm)

<br>

## Section 1: Creation and export of the flight paths

**Step 1:** Install the Map Pilot Pro in your phone either through your mobile app store (Google Play Store/Apple App Store etc) or install it using an apk file.

**Step 2:** Open the Map Pilot Pro app in your phone.

<img src="https://github.com/user-attachments/assets/692b6d9e-b59a-4da5-9de8-18d5f6e6cfe8" alt="downloaded_app" width="225" height="450"> <br>

**Step 3:** Click on 'Settings'.

<img src="https://github.com/user-attachments/assets/fa697bed-0643-437e-8bf4-155668b20d33" alt="map_pilot_pro_screen" width="500" height="225"> <br>

**Step 4:** Go to 'Advanced Settings' and turn on "Export Simple KMZ"

<img src="https://github.com/user-attachments/assets/d7ec2ef1-414a-4774-abc0-2fe10babe22a" alt="advanced_settings" width="500" height="225"> <br>

Without this, you will only be able to export the flight paths in the mme (maps made easy) format, a non-interoperable file format not recognised by any other GIS or drone app.

**Step 5:** Go to 'Flight Camera' > 'Model'> and then select a camera which has an "Export Only" text given in the bracket. This is also necessary to export the flight path as a kmz file.

<img src="https://github.com/user-attachments/assets/81478e53-26e1-4cb0-a9c7-64de673d97e4" alt="flight_camera_model" width="500" height="225"> 
<img src="https://github.com/user-attachments/assets/6b624b82-8b18-48f3-a397-0a84a6c9b26e" alt="flight_camera_model" width="500" height="225"> <br> <br>

**Step 6:** *[Start from step 18 if you want to upload and use boundary files of your region of interest as reference for drawing the flight paths]* Go back to the home screen and click on 'New Mission'

<img src="https://github.com/user-attachments/assets/fa697bed-0643-437e-8bf4-155668b20d33" alt="map_pilot_pro_screen" width="500" height="225"> <br>

**Step 7:** Click 'OK' on both the pop-ups

<img src="https://github.com/user-attachments/assets/13f593b0-dbba-4605-878b-0fa0e30025e9" alt="pop_up_1" width="500" height="225"> 
<img src="https://github.com/user-attachments/assets/4149b588-dd0e-4253-a2b9-cb26570a1ca2" alt="pop_up_2" width="500" height="225"> <br> <br>

**Step 8:** Move the viewport (the region of the map the app shows to you) to the desired flight path region.

By default, the location in the viewport will be Lower Manhattan. If your drone flight location is close to where you're right now, you can turn on your phone GPS and it will automatically move to that area.
If, not zoom out of Lower Manhattan and zoom into your region of interest.

<img src="https://github.com/user-attachments/assets/f6a662a2-80d1-467c-9971-58c1a3f4d233" alt="default_viewport" width="500" height="225"> 
<img src="https://github.com/user-attachments/assets/aaf5bd08-a6fb-41ec-8aab-9a1dcff34862" alt="zoomed_out_viewport" width="500" height="225"> 
<img src="https://github.com/user-attachments/assets/28fb3231-44aa-4aa4-beba-4ce98da2424d" alt="zoomed_in_roi" width="500" height="225"> <br> <br>

**Step 9:** Specify the altitude (80 m is ideal), speed (slow speeds of 3 m/s is recommended for clear capture of images), and overlap (80% overlap in the front and sides (80/80) is recommended. For class specific recommendations, you can visit read [this article][1]).Click on 'Norm Mission' and then 'Normal'. 

<img src="https://github.com/user-attachments/assets/21a01ea1-84cc-4078-92fc-60c043bfb61e" alt="norm_mission" width="500" height="225"> <br> 

**Step 10:** Click on the bar with 'export only' to minimize the top bar so that you have more visibility of your region of interest.

<img src="https://github.com/user-attachments/assets/fa676f90-b1e0-4caf-9ab7-cd5451983721" alt="minimize" width="500" height="225"> <br> 

**Step 11:** Long press on a portion of the screen to set your purple "Home point" dot. This is the point to where the drone will return to after the flight path has been completed.

<img src="https://github.com/user-attachments/assets/94502011-f412-4589-a537-8128d8d7b9ff" alt="home_dot" width="500" height="225"> <br> 

**Step 12:** Long press on a portion of the screen to set your orange "Boundary" dots to cover the entire survey area. 

There will be three dots in the beginning. Use these three dots to cover the entirety of the desired survey area in the first pass. After this, use the smaller orange dots to more snuggly fit the survey area. Be careful whil moving the smaller dots to avoid messing up as there is no undo or redo option. If you mess up, delete the flight path by using the dust bin icon on the top left.

<img src="https://github.com/user-attachments/assets/9ffdd5c4-53c6-48e9-b97e-7a617a0a27c4" alt="boundary_dot" width="500" height="225"> <br> 

**(optional) Step 13:** If you want this flight plan to be accessible later on, click on the floppy disk icon on the top left, name it and save the mission/flight path. 

<img src="https://github.com/user-attachments/assets/536def51-7930-44b8-89c2-0d924cc9727a" alt="rename" width="500" height="225">
<img src="https://github.com/user-attachments/assets/552b49f1-b056-4cb2-9cfa-b738aebd8c41" alt="renamed" width="500" height="225"> <br> <br>

**Step 14:** Click on the drone/airplane icon on the top right and click on the 'Export' icon

<img src="https://github.com/user-attachments/assets/7d956515-797e-4cb0-bd72-4ba52b86545e" alt="drone_icon" width="500" height="225">
<img src="https://github.com/user-attachments/assets/b86194ae-5043-44f1-8ca8-1db8cfb20dd5" alt="export" width="500" height="225"> <br> <br>

**Step 15:** Click on 'No' for the 'Terrain Awareness' pop-up, as this option is not available in the free version.

<img src="https://github.com/user-attachments/assets/08a2a011-0e18-4aa8-906d-143e589689d4" alt="pop_up_terrain_awareness" width="500" height="225"><br>

**Step 16:** Give a name for the flight path, click on 'Export' and press 'OK' on the next popup.

<img src="https://github.com/user-attachments/assets/844db279-6cef-40d0-a2a8-10560661ad89" alt="rename" width="500" height="225">
<img src="https://github.com/user-attachments/assets/b43fe903-185f-4ddf-94bf-54a6ca21887f" alt="export" width="500" height="225"> <br> <br>

You can now find the downloaded kmz file for use in the drones, in the 'Downloads' folder of your phone. 

**Step 17:** Click on left facing triangle which will reveal the 'Exit' icon. Click and exit to the Home screen.

<img src="https://github.com/user-attachments/assets/17ae4c17-33ca-4648-9047-16d983e650d2" alt="exit" width="500" height="225"> <br>

**(optional) Step 18:** Click on 'File Manager'

If you need to import a boundary file for your region(s) of interest like a shapefile, then you first need to convert it to a kmz format (using tools like QGIS) and keep it ready in the device or google drive.

<img src="https://github.com/user-attachments/assets/fa697bed-0643-437e-8bf4-155668b20d33" alt="file_manager" width="500" height="225"> <br>

**(optional) Step 19:** Click on 'Import'

<img src="https://github.com/user-attachments/assets/e129a13d-61d3-4c66-a441-585bfe4e0d9c" alt="import" width="500" height="225"> <br>

**(optional) Step 20:** Navigate to the location in your device or google drive to find the saved boundary file, and then click on it.

<img src="https://github.com/user-attachments/assets/7e90eb4b-1a06-4952-9af1-468c46288717" alt="drive" width="225" height="450">
<img src="https://github.com/user-attachments/assets/6143fbfc-6b30-4078-894b-0c661e6c0bc7" alt="search" width="225" height="450">
<img src="https://github.com/user-attachments/assets/6feef804-5f7b-4039-8aaa-521e731b9a99" alt="find" width="225" height="450"> <br> <br>

**(optional) Step 21:** Click on 'Imported'

<img src="https://github.com/user-attachments/assets/3fd7872a-cbfb-4539-a987-6683560dda27" alt="imported" width="500" height="225"> <br>

**(optional) Step 22:** Under 'Imported Overlay Files', turn on the 'Show KML'

<img src="https://github.com/user-attachments/assets/bc475eee-34fe-4930-bc04-171ebb8d7fe7" alt="import" width="500" height="225"> <br>

**(optional) Step 23:** Click on 'Sync'

This is needed so that the boundary files show up in the map.

<img src="https://github.com/user-attachments/assets/a37ae6af-85ea-4f93-b1a0-ff2acf991cd7" alt="sync" width="500" height="225"> <br>

**(optional) Step 24:** Under 'Imported Overlay Files', turn on the 'Show KML'

<img src="https://github.com/user-attachments/assets/bc475eee-34fe-4930-bc04-171ebb8d7fe7" alt="show_kml" width="500" height="225"> <br>

**(optional) Step 25:** Repeat Step 6 to 16 but with the boundary files as a guide for drawing the flight paths.

You can see that in this case, there are grey line segments after the white line segments in the flight path. The grey lines indicate whose paths which cannot be completed in one flight,
and multiple batteries would be required to complete such large survey areas. The drone will return to the home point after reaching the end of the white line, and then after battery has been replaced,
it will continue from where it left off. For details on how much distance the flight path covers, the number of batteries required, area covered, number of images captured, amount of space it will take in
storage, click on the polygon icon on the left of the screen. If you have a paid account and wants to have better terrain awareness for the flight path, click on the icon on the bottom left with one coniferous 
tree and two  snow-capped mountains. It will allow you to select the digital elevation model. ASTER is the worst among them for most regions. SRTM is better, but Copernicus is likely the most accurate among 
these three for most regions, especially the mountainous regions.


<img src="https://github.com/user-attachments/assets/bc32b989-9ea4-40da-8865-b196c01b3a64" alt="boundary_files" width="500" height="225">
<img src="https://github.com/user-attachments/assets/e7a38cb7-83b2-4cc8-b50e-3a72ce70b0a4" alt="white_and_grey_lines" width="500" height="225">
<img src="https://github.com/user-attachments/assets/facc8bfc-35cf-4a65-8c5d-88302e5450f5" alt="more_details" width="500" height="225"> <br> <br>

## Section 2: Uploading the flight path to the drone or drone app

### For DJI Mini 5 Pro

For DJI Mini 5 Pro, the kmz file must be uploaded to the remote controller. But there is no straightfoward way to do this. Rohan Baishya explains in this section, a foolproof method to do this (based on [this tutorial][2].

**Step 1:** In the RC/DJI fly app, connect the drone and press 'GO FLY' 

NOTE: Make sure the drone is placed outside for adequate satellite signal

<img src="https://github.com/user-attachments/assets/e190f2ad-b712-4e35-8457-388bdc85222f" alt="rc-dji-app" width="600" height="325"> <br>

**Step 2:** Click on the map at the bottom left corner (make sure your position is being shown on the map)

<img src="https://github.com/user-attachments/assets/a60b004d-0a62-4db1-9c0e-385d0064d2c4" alt="map" width="600" height="325"> <br>

**Step 3:** Click on the waypoint icon highlighted in yellow on the left-centre, and create a waypoint with at least 2 points.

<img src="https://github.com/user-attachments/assets/03b0d842-8721-4b8c-9ae2-5e0a778ae37b" alt="waypoint-icon" width="600" height="325"> <br>

**Step 4:** Click on the note/paper icon to left of the text 'Waypoint'. 

<img src="https://github.com/user-attachments/assets/03b0d842-8721-4b8c-9ae2-5e0a778ae37b" alt="waypoint-icon" width="600" height="325"> <br>

**Step 5:** Select the waypoint created and click on 'SAVE' > 'Save'.

<img src="https://github.com/user-attachments/assets/ac1487d9-9e6d-4e17-bad9-adae3d681d54" alt="SAVE" width="600" height="325">
<img src="https://github.com/user-attachments/assets/e7328b47-2fb6-4d4d-8b96-77817e308da5" alt="Save" width="600" height="325"> <br><br>

**Step 6:** Return to the home screen by clicking on the left-facing arrow on top-left and press on 'Save and Exit'. 

<img src="https://github.com/user-attachments/assets/d929a78b-f3f3-4b77-8dad-99bc6d8fb18e" alt="save_and-exit" width="600" height="325"> <br>

**Step 7:** Connect the RC to your desktop/laptop (using an USB or any other means available) and follow this path
```
DJI RC 2 > Internal Shared Storage > Android > data > Dji.go.v5 > files > waypoint
```
<img src="https://github.com/user-attachments/assets/3587bd4a-abd0-46f4-aff8-5bf99097714a" alt="path" width="800" height="525">

Once you reach the waypoint folder, select the most recently created folder. This will contain the waypoint created in the steps above.

**Step 8:** Open the folder and copy the name of the KMZ file. 

<img src="https://github.com/user-attachments/assets/846c81e1-0beb-4db2-9d58-6e7ad837b587" alt="copy_name" width="800" height="525"> <br>

**Step 9:**  In the PC, change the name of the KMZ file generated from Map Pilot Pro app to the copied name. In this case the kmz file named "Pasighat_test_1.kmz" is changed 
to "90A8341C-CFE2-4E84-8681-B5B081E5C815.kmz"

<img src="https://github.com/user-attachments/assets/1c6e17c9-8c26-4079-8859-3410e3b3ebc5" alt="rename" width="800" height="525"> <br>

**Step 10:**  Copy it to the folder we reached in step 7, and replace the original waypoint kmz file created in the RC/DJI Fly app with the kmz file 
created in Map Pilot Pro app. Now you can disconnect the RC from the PC.

<img src="https://github.com/user-attachments/assets/cbe6fc64-53fd-4dc1-864b-2cc598656890" alt="replace_original" width="800" height="525"> <br>

**Step 11:**  Now connect the RC to the drone. 

**Step 12:**  Once the drone is connected, navigate to the note/paper icon to the left of the text 'Waypoint'. 

<img src="https://github.com/user-attachments/assets/ddc9ed80-da2e-4eb3-b2d8-726252e020ad" alt="note_icon" width="600" height="325">
<img src="https://github.com/user-attachments/assets/abac377f-17dd-476c-8d81-6ad8b4fb2e34" alt="note_icon" width="600" height="325"> <br>

The kmz file created in Map Pilot Pro should be visible now in the saved waypoints window. 


## Section 3: Flying the drone

### DJI Mini 4 pro

**Step 1:** Connect the RC with the drone and enter an open ground

**Step 2:** Select this option — it is the waypoint mode.

<img src="images/step1.jpeg" width="600"> <br>

**Step 3:** Select this list button.

<img src="images/step2.jpeg" width="600"> <br>

**Step 4:** Select the path you want to operate on as uploaded previously on DJI RC.

<img src="images/step3.jpeg" width="600"> <br>

**Step 5:** It will appear like this showing the number of waypoints present. Then press **Next**.

<img src="images/step4.jpeg" width="600"> <br>

**Step 6:** All the fields will already be filled as per the data configured in Map Pilot Pro. Press **Go** to make the drone fly on the selected flight path.

<img src="images/step5.jpeg" width="600"> <br>

## Section 4: Creation of orthomosaic in WebODM

WebODM is a powerful free and open-source software for the rendering of orthomosaics as well as 3D imagery from the drone photographs. It is very advanced with a lot of options to play around with to efficiently and quickly process large number of drone imagery.
The bottleneck for processing would be the system RAM size, availability of GPU, disk space etc. Higher the RAM size, availability of good GPUs for GPU acceleration, and sufficient disk space for storage of input/output files as well as intermediate products (there is an option to delete intermediate products in WebODM while processing, for those lacking space). 

**Step 1:** Install WebODM and dependencies

WebODM works best in Linux, and also in Windows and Apple and Operating Systems (OSes). The installation procedure for all the OSes is described [here][3]. So, I am not repeating it here. Git and Docker would need to be installed for the installation of WebODM. If you have trouble installing WSL2 (Windows Subsystem for Linux 2, which is needed for running the Linux kernal within Windows), then refer to installation guide along with troubleshooting help for WSL2 [here][4].

**Step 2:** Optimise the functioning of WebODM

If using Windows, you can allocate more memory by searching for WSL Settings...

<img src="https://github.com/user-attachments/assets/2e567a29-68eb-40b3-a284-6cb4d1ca7190" alt="search_wsl_settings" width="800" height="750">

...and editing the 'Memory Size' option. 28,000 MB here is nearly 28 GB (out of the 32 GB RAM available in the system). Leave some memory for Windows.

<img src="https://github.com/user-attachments/assets/29cda4bd-4aed-40b3-93bc-d2f81d8f8f03" alt="search_wsl_settings" width="800" height="525"> <br>

**Step 3:** Launch WebODM

Once you follow the instructions for installation, you can launch WebODM from the Bash/Git Bash/other command line interface and open WebODM in a browser at "http://localhost:xxxx/". 
Make sure you have Docker (Dekstop) running before launcing WebODM as it will not launch successfully without it.

**Step 4:** Create a WebODM account

Create an account. You can log in and log out any time. If you close the browser without logging out, you will be logged in and won't have to log in again the next time you launch WebODM.

**Step 5:** Click on 'Select Images and GCP'

<img width="1907" height="427" alt="image" src="https://github.com/user-attachments/assets/bea593f3-d275-48c4-abaf-a4391e05ccca" /> <br>

**Step 6:** Navigate to the folder containing all the images from a flight path, select all of them, and 'Open'

<img width="1342" height="668" alt="image" src="https://github.com/user-attachments/assets/0f69ad78-c530-4a9f-91e4-b4df646f1762" />

**Step 7:** Edit the parameters as required before starting processing, by clicking on 'Edit' under 'Options'

<img width="1616" height="781" alt="image" src="https://github.com/user-attachments/assets/80f47048-6b35-4260-8938-36664c0571e8" />

I have found that 'Fast Orthophoto' was surprisingly consistently providing orthomosaics/orthophotos with less distortions than when it is not used.

<img src="https://github.com/user-attachments/assets/7498ab56-f985-4ff4-94ca-b1fdfd7b8d89" alt="downloaded_app" width="450" height="900"> <br>

If you enable 'skip-3dmodel', it saves a good amount of time (especially on not-so-powerful computers), and it doesn't seem to effect the the quality of the orthomosaics either way.

<img src="https://github.com/user-attachments/assets/3dc1b139-953f-4538-9f56-31006cf3921f9" alt="downloaded_app" width="450" height="900"> <br>

If you are facing memory issues while processing, you can reduce the maximum concurrency which will reduce the multi-tasking (and hence increase the time taken).
Enabling optimise-disk-space deletes intermediate files to optimise the disk space usage (if the system is low on disk space).

<img src="https://github.com/user-attachments/assets/3178f3b6-c6f9-475e-a3e2-ea66cc4ad84e" width="450" height="900"> <br>

**Step 8:** Click on 'Review' and 'Start Processing' (make sure you're not running other computationally heavy processes like using QGIS or having many tabs open in the browser)

<img width="1616" height="781" alt="image" src="https://github.com/user-attachments/assets/80f47048-6b35-4260-8938-36664c0571e8" />

<img width="1603" height="764" alt="Screenshot 2026-03-11 202144" src="https://github.com/user-attachments/assets/b7d5926e-9ca4-4e63-9653-cec1f1edf466" /> <br>

It will show a blue loading bar which shows the progress of uploading the images

<img width="1621" height="319" alt="image" src="https://github.com/user-attachments/assets/eff8e8f6-dabe-4c2e-8618-975c80651071" />

Once uploaded, the task starts. You can tunr 'On' the 'Task Output:' to see each and every step in real time.

<img width="1624" height="565" alt="image" src="https://github.com/user-attachments/assets/7f3c48a1-ddaa-4f18-ba92-53ae3e0f1386" /> <br>

**Step 9:** View the orthomosaic by clicking on the imagery on the screen.

<img width="1586" height="478" alt="image" src="https://github.com/user-attachments/assets/174198d7-407e-45e7-b8e4-6c5f2fcd46d8" /> <br>

Here you can view the 2D and 3D imagery generated.

You can change the basemap between many options, by clicking on the four squares icon.

<img width="1651" height="901" alt="image" src="https://github.com/user-attachments/assets/eab1e034-f332-427c-8a5a-bc15631f6d22" />

<img width="374" height="573" alt="image" src="https://github.com/user-attachments/assets/ffc6e0e0-15b1-481b-b3fc-77cb9c941141" />

There are several other options available in this viewing screen. You can explore all of them. <br>

**Step 10:** Download the necessary files. 

You can download the orthophoto, quality report, point cloud (for 3D imagery) etc from the 'Download' tab. The 'Backup' option downloads everything in a zip file.

<img width="1603" height="536" alt="image" src="https://github.com/user-attachments/assets/81245006-1bfc-402b-a071-ece9492bbc50" />

You can download the data log (Task Output log on the console) by clicking on the download icon underneath the 'Task Output' console. This will be useful for troubleshooting any issues.



















[1]: https://docs.webodm.net/getting-started/image-capture/
[2]: https://youtu.be/E5GnutgGH2c?si=ZaDjesyPdX-LCRWC
[3]: https://docs.webodm.org/installation/#docker
[4]: https://github.com/paulvpop/gis-land-cover-mapping/blob/main/01.%20Software%20installation.md#52-podman-and-windows-subsystem-for-linux-2-use-this-for-windows







