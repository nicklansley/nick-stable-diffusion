# Nick's Stable Diffusion Playground

This project is a clone of https://github.com/CompVis/stable-diffusion.git
 
I've built a docker-based web service around the original project, which incorporates a few extra features, and assumes you are running Docker Desktop over WSL2 on Windows 10/11.

* Model-Always-Loaded backend server means that incoming requests go straight to creating images rather than model-loading.
* Redis-based scheduler and queue management of prompts, so the backend only processes one prompt at a time.
* Simple non-framework UI that can be adapted as desired. The UI looks 'early 1990s' right now but it does its job.
* A simple API called by the JavaScript in the UI to send prompt requests, check the queue and see the library of results
* docker compose volumes can be adjusted to save the pretrained image model, caches and output library of images on a disk outside of Docker.
* The backend is written in Python and the UI is written in JavaScript.
* Output images are in PNG format.
* New '/advanced.html' with access to use input images, lock in the seed value, image size, DDIM Steps, Scale and Downsampling Factor. Bear in mind some settings can cause errors on the backend, so watch the backend server logs should your request disappear from the queue almost immediately with no results.
* A stupid prompt inspiration page generating random and daft prompts. A good idea at the time.

## 8 steps to Fast-start
1. Make sure you have an NVidia graphics card and a NVidia's GPU driver installed. This is how the backend will render the images.
2. The graphics card needs to have at least 10 GB of GPU memory in total. I use an RTX 3090 Ti but should work on 3080s and 2080s.
3. You should already be using Docker Desktop in WSL2 for all kinds of reasons including performance, 
but by default WSL2 does not have the 'right' to use maximum memory, and Docker reports that this project uses over 20GB memory at the present time. To overcome the max memory error, open (or create)
a file in your Windows home directory called <b>.wslconfig</b> and put a 'memory=' property in that file with a memory size of 4GB lower than your PC's memory (mine has 64 GB).
<pre>
[wsl2]
memory=60GB 
</pre>
4. Read docker-compose.yml and adjust the two volumes to your needs - On my S: hard drive I have created a folder 'nick-stable-diffusion-data' and then created three empty sub-folders: 'cache', 'model' and 'library'.
Docker will connect these physical locations to the internal volumes of the containers. 
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
7. At first start the backend will then download another 2.73 GB of data which it will store in the 'cache' folder you set up on your drive and set up in docker compose.

10. You can now start the UI by going to this web address on your web browser: <pre>http://localhost:8000</pre>


## Notes
The UI is super-simple and is designed to be easily adapted to your needs. I've avoided frameworks so that you can add yours as needed. I've written the JavaScript in a spaced out and 'pythonic' manner so it is easy to read (I hope!)

I've written the scheduler and frontend web server in Python. The Scheduler uses a simple FIFO queue to manage the prompts with Redis as the queuing database. I've used a class based on BaseHTTPRequestHandler to handle the requests from the UI.


### Library Page
The UI includes a library page where you can view the images created so far. If you want to empty the library, simply go to the 'library' folder you created in 'fast start' step 4 and delete everything in it.
If you want to delete a specific image, double-click it on the library page, and select 'OK' to the alert prompt.

The library page is useful for observing how many seconds it took to generate each image, as it is displayed above each group of images. My PC always has it at around 4 secs/image. If yours is different, 
you can adjust the value in the JavaScript at the top of index.js - change the very first line - const SECS_PER_IMAGE = 4; - to the number of seconds per image you are experiencing.
This will make the countdown on the UI more accurate when waiting for your prompt to be processed.

## API
The API is a simple RESTful API that can be used by the UI to send requests to the backend.
I will document it here but for a quick glance, look at the go() function in the frontend's index.js file and see its fetch() call.
Good luck setting this up on your PC - let me know how you get on.

## NSFW
Note that I have removed the safety catch and allow this project to create any image it desires. But doing so comes with great responsibility and you must
not forget what you agreed to in step 5 above.
If you prefer to avoid NSFW content, re-enable the safety catch in the backend code by changing server.py line 237 from calling check_safety() to calling orig_check_safety()
