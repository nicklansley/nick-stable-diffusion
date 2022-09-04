# Nick's Stable Diffusion Playground

This project is a clone of https://github.com/CompVis/stable-diffusion.git
 
I've built a docker-based web service around the original project, which incorporates a few extra features, and assumes you are running Docker Desktop over WSL2 on Windows 10/11.

- Model-Always-Loaded backend server means that incoming requests go straight to creating images rather than model-loading.
- Redis-based scheduler and queue management of prompts, so the backend only processes one prompt at a time.
- Simple non-framework UI that can be adapted as desired. The UI looks 'early 1990s' right now but it does its job.
- A simple API called by the JavaScript in the UI to send prompt requests, check the queue and see the library of results.
- docker compose volumes can be adjusted to save the pretrained image model, caches and output library of images on a disk outside of Docker.
- NEW: Manipulate images you have already created that reside in the library. Click on an image in the library and a new advanced page will open in a new tab with all the settings (seed, ddim_steps, scale, image link etc) used to create that image, so you can easily make variations by adjusting the controls.
- NEW: (Advanced Page) Link to any image on the web and use that for the input image to the AI. You can adjust how much the image can be changed from 0.01% (no change) to 100% (no original image). You can also choose any image in the library folder on your machine (just move it there if necessary). The only condition is that the image can be retrieved without login and the website at the far end won't mind Python's urllib library pulling the image. Images are resized internally to 512x512px without affecting the original image's aspect ratio (black bands will appear on the top/bottom or left/right side of the longer edge to make it square). This aids the AI as it is finicky about sizes and size ratios.
- The 'advanced.html' page gives you creative access to use input images, lock in the seed value, image size, DDIM Steps, Scale and Downsampling Factor. Bear in mind some settings can cause errors on the backend, so watch the backend server logs should your request disappear from the queue almost immediately with no results.
- The backend is written in Python and the UI is written in JavaScript.
- Output images are in PNG format so don't suffer from JPEG artifacts.
- NEW: 'dark mode' button for all the web pages, for those late-night creative AI sessions without being blasted by light!
- A stupid prompt inspiration page generating random and daft prompts. A good idea at the time.

## 8 steps to Fast-start
1. Make sure you have an NVidia graphics card and a NVidia's GPU driver installed. This is how the backend will render the images.
2. The graphics card needs to have at least 10 GB of GPU memory in total. I use an RTX 3090 Ti but should work on 3080s and 2080s.
3. You should already be using Docker Desktop in WSL2 for all kinds of reasons including performance, 
but by default WSL2 does not have the 'right' to use maximum memory, and Docker reports that this project uses over 20GB memory at the present time. To overcome the max memory error, open (or create)
a file in your Windows home directory called   <b>.wslconfig</b> for example on my PC: <pre>C:\Users\nick\\.wslconfig</pre> and put a 'memory=' property in that file with a memory size of 4GB lower than your PC's memory (mine has 64 GB).
This does not mean that WSL2 will grab all but 4GB of your PC memory, it's just that you are giving it the 'right' to use that much if it really needs it.
Why 4GB? I am guessing Windows can just about keep running on 4GB of memory, but I have not tested it...! The file should look like this:
<pre>   
[wsl2]
memory=60GB 
</pre>
4. Read docker-compose.yml and adjust the three volumes to your needs - On my S: hard drive I have created a folder 'nick-stable-diffusion-data' and then created three empty sub-folders: 'cache', 'model' and 'library'.
Docker will connect these physical locations to the internal volumes of the containers. You can change the 'device:' values to folders on your PC you would prefer the containers to use.
<pre>
volumes:
  app-cache-on-s:
    driver: local
    driver_opts:
      o: bind
      type: none
      device: S:\nick-stable-diffusion-data\cache
  model-on-s:
    driver: local
    driver_opts:
      o: bind
      type: none
      device: S:\nick-stable-diffusion-data\model
  library-on-s:
    driver: local
    driver_opts:
      o: bind
      type: none
      device: S:\nick-stable-diffusion-data\library
</pre>


5. You will need to download the Stable Diffusion model - at the time of writing I am using v1.4 of the model. 
Go here https://huggingface.co/CompVis/stable-diffusion-v-1-4-original and create an account. Then go back to this page to accept the terms.
Then go here https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt 
and download this file. 
6. Copy the downloaded file to the 'model' folder you have setup in step 4. Note how I use 'S:\nick-stable-diffusion-data\model'.
You will need to place the .ckpt file there and rename it <b>model.ckpt</b> .

6. Run docker-compose to build the project then start the backend, scheduler and frontend. Downloading the container images is a one-time operation but takes time and several GB of download!
I am happy to accept that right now the backend build is a bit overkill - it even compiles some libraries from rust! I'll clean it up later.
<pre>
docker compose up -d --build 
</pre>
7. At first start the backend will then download another 2.73 GB of data 
which it will store in the 'cache' folder you set up on your drive and set up in docker compose. 
THis will only happen the first time - it doesn't do this every time you start the backend.
The second and subsequent times you start the backend it will be live within about 30 seconds, sometimes sooner.
You can't use the application until you see, in the backend server log:

<pre>
Backend Server ready for processing on port 8080
</pre>

8. You can now start the UI by going to this web address on your web browser: <pre>http://localhost:8000</pre> - that's port 8000


## Notes
The UI is super-simple and is designed to be easily adapted to your needs. I've avoided frameworks so that you can add yours as needed. I've written the JavaScript in a spaced out and 'pythonic' manner so it is easy to read (I hope!)

I've written the scheduler and frontend web server in Python. The Scheduler uses a simple FIFO queue to manage the prompts with Redis as the queuing database. I've used a class based on BaseHTTPRequestHandler to handle the requests from the UI.

### Home page
This page enables you to type in a prompt, choose the number of images you wish to create in groups of 3 from 3 to 30, and set the AI processing!
It's designed to be simple to use, with a random seed value generated for you. 

Type in a prompt, set the number of images (from 3 to 30 in steps of 3) and 'click to send request'.
The request will be queued and the scheduler will pick it up and process it. The scheduler will then send the results back to the frontend web server which will then send the results back to the web page.

Once the images appear, hover over them with your mouse to make them much bigger. You can drag them onto your desktop with your mouse to save them outside the library.

These images belong to you. They are your copyright. You can use them for whatever you wish.   

If you don't want to wait, you can send several requests - just keep clicking the 'click to send request' button.
Your requests will be added to the queue. Note that only the last request you sent will result in images appearing on the page,
but you can see all the requests as each completes on the library page (see below).

If you wish, you can use a service such as ngrok to get your computer a temporary web address on the internet, then your friends and you can
all submit images using the same page, they will just get added to the queue and processed in turn. Indeed, the whole application is designed for multi-user operation.


Behind the scenes for this simple prompting page, the ddim_steps is set to 50, scale is set to 7.5, image size is 512x512px and the downsampling factor is 8.
These defaults can be altered using the capital-letter variables at the top of script file  backend-sd-server/server.py if you wish to change them,
 but these seem to work best and are suggested by the authors of the model. Use the advanced page to override them for moe control in a session.

### Library Page
The UI includes a library page where you can view the images created so far. 
Images are grouped by prompt and date/time with the latest at the top of the page.
The group header also displays the various settings that either the AI used, or you used if you prompted vi the advanced page.

Check a box at the top of the page to allow it to refresh itself every 10 seconds.

The library page has been updated to improve the user experience with images:
* Hover over an image to enlarge it.
* Drag image to your desktop to save it outside the libray folders.
* Click to open a new advanced page with the settings that made this image already preset so you can maipulate it further.
* Right-click to delete the image from the library (agree 'OK' to confirm).

If you want to empty the library, simply go to the 'library' folder you created in 'fast start' step 4 and delete everything in it.
If you want to delete a specific image, right-click it in the library page, and select 'OK' to the alert prompt.

The library page is also useful for observing how many seconds it took to generate each image, as it is displayed above each group of images. My PC always has it at around 4 secs/image. If yours is different, 
you can adjust the value in the JavaScript at the top of index.js - change the very first line - const SECS_PER_IMAGE = 4; - to the number of seconds per image you are experiencing.
This will make the countdown on the UI more accurate when waiting for your prompt to be processed.

### Advanced Page
The Advanced page allows you to set the parameters for the AI to use when generating the image.
You can use this page to link to a source image on the web, or in your library folder. 

If you specify an image, it will be included as the first image '00-original.png' in the output.
<i>Note that you do not have copyright over the original image, only the AI-created images.</i>

If you click an image in the library, it will open a new advanced page with the settings that made that image already preset so you can maipulate it further.


## API
The API is a simple RESTful API that can be used by the UI to send requests to the backend.
I will document it here but for a quick glance, look at the go() function in the frontend's index.js file and see its fetch() call.
Good luck setting this up on your PC - let me know how you get on.

## Safety Catch
Note that I have disabled the safety catch and allow this project to create any image it desires. But doing so comes with great responsibility and you must
not forget what you agreed to in step 5 above. I only disabled the safety catch because Rick Astley (the NSFW replacement image) was appearing too often on innocent prompts! 
If you prefer to avoid NSFW content, re-enable the safety catch in the backend code by changing server.py line 237 from calling check_safety() to calling orig_check_safety()

## Licenses
The favicon used by this application was generated using the following graphics from Twitter Twemoji:
- Graphics Title: 1f929.svg
- Graphics Author: Copyright 2020 Twitter, Inc and other contributors (https://github.com/twitter/twemoji)
- Graphics Source: https://github.com/twitter/twemoji/blob/master/assets/svg/1f929.svg
- Graphics License: CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)