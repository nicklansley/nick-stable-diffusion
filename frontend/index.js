
const SECS_PER_IMAGE = 4; // depends on GPU image creation speed - 4 works well for me
let global_currentQueueId = '';
let global_imagesRequested = 0;
let global_countdownValue = 0;
let global_countdownTimerIntervalId = null;

/**
 * Send the text prompt to the AI and get a queue_id back in 'queue_id' which will be used to track the request.
 * @returns {Promise<void>
 */
const go = async () =>
{
    document.getElementById('status').innerText = "Creating images..."
    document.getElementById('buttonGo').innerText = "Creating images...";
    document.getElementById('buttonGo').enabled = false;

    global_imagesRequested = parseInt(document.getElementById('num_images').value);

    //If the timer is already running and the button is clicked again, the user is sending a new request
    //to be added to the queue. So we need to stop the countdown timer, as the countdown is no longer
    //applicable to this new current request.
    if(global_countdownTimerIntervalId)
    {
        clearInterval(global_countdownTimerIntervalId);
        global_countdownTimerIntervalId = null;
    }

    const data = prepareRequestData();
    const rawResponse = await sendPromptRequest(data);
    await processPromptRequestResponse(rawResponse);
}


const prepareRequestData = () =>
{
    const data = {
        prompt: document.getElementById("prompt").value,
        num_images: global_imagesRequested,
        seed: 0
    }
    if (document.body.innerText.includes("DDIM Steps:"))
    {
        //We have the advanced options incoming for the request from advanced.html
        data['seed'] = parseInt(document.getElementById("seed").value);
        if (data['seed'] === '')
        {
            data['seed'] = 0;
        }

        data['height'] = parseInt(document.getElementById("height").value);
        if (data['height'] === '')
        {
            data['height'] = 512;
        }

        data['width'] = parseInt(document.getElementById("width").value);
        if (data['width'] === '')
        {
            data['width'] = 512;
        }

        data['min_ddim_steps'] = parseInt(document.getElementById("min_ddim_steps").value);
        data['max_ddim_steps'] = parseInt(document.getElementById("max_ddim_steps").value);

        data['scale'] = parseFloat(document.getElementById("scale").value);
        if (data['scale'] === '')
        {
            data['scale'] = 7.5;
        }

        data['original_image_path'] = document.getElementById("original_image_path").value;

        // Note that the strength slider represents a value fomo 0.01 - 99.9% whereas strength is a float from 0.0 to 0.999
        // I used percentage because it is easier to understand for the user.
        data['strength'] = parseFloat(document.getElementById("strength").value) / 100;
        if (data['strength'] === '')
        {
            data['strength'] = 0.75;
        }


        let downsamplingFactor = 0;
        const dsfRadioGroup = document.getElementsByName("downsampling_factor");
        for(let i = 0; i < dsfRadioGroup.length; i++)
        {
            if (dsfRadioGroup[i].checked)
            {
                downsamplingFactor = dsfRadioGroup[i].value;
                break;
            }
        }
        data['downsampling_factor'] = downsamplingFactor;
    }
    return data;
}

const sendPromptRequest = async (data) =>
{
    document.getElementById("output").innerText = "";

    const rawResponse = await fetch('/prompt', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    return rawResponse;
}

const processPromptRequestResponse = async (rawResponse) =>
{
    if (rawResponse.status === 200)
    {
        const queueConfirmation = await rawResponse.json();

        global_currentQueueId = queueConfirmation.queue_id;
        document.getElementById('status').innerText = "Request queued - check the queue for position";
    }
    else
    {
        document.getElementById('status').innerText = `Stable Diffusion Engine Status: Sorry, an HTTP error ${rawResponse.status} occurred - have another go!`;
    }
    document.getElementById('buttonGo').innerText = "Click to send request";
    document.getElementById('buttonGo').enabled = true;

    await createImagePlaceHolders();
}


const getImageList = async () =>
{
    const imageListResponse = await fetch('/imagelist', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({queue_id: global_currentQueueId})
    });

    if (imageListResponse.status === 200)
    {
        return await imageListResponse.json();

    }
    else
    {
        return { 'completed': true, 'images': [] };  // error condition
    }


}


/**
 * Display the queue.
 * @param queueList
 * @returns {Promise<void>}
 */
const displayQueue = async (queueList) =>
{
    let backendProcessingRequestNow = false;

    const queueUI = document.getElementById("queue");
    if (queueList.length === 0)
    {
        queueUI.innerHTML = "Current queue: Empty<br>You'll be first if you submit a request!";
    }
    else
    {
        // Is my request being currently processed? If it's first in the queue, then yes.
        backendProcessingRequestNow = queueList[0].queue_id === global_currentQueueId

        const maxDDIMSteps = queueList[0].max_ddim_steps ? queueList[0].max_ddim_steps : 0;
        const minDDIMSteps = queueList[0].min_ddim_steps ? queueList[0].min_ddim_steps : 0;

        const imageRequestCount = queueList[0].num_images * (maxDDIMSteps - minDDIMSteps + 1);

        // The first item in the queue is the one that the AI is currently processing:
        queueUI.innerHTML = `<p><b>Now creating ${imageRequestCount} image${imageRequestCount > 1 ? "s" : ""} for${backendProcessingRequestNow ? " your request" : " "}:<br>'${queueList[0].prompt}'...</b></p><br>Current queue:<br>`;

        const processingDiv = document.createElement("div");
        processingDiv.innerHTML = `<b>Now creating ${imageRequestCount} image${imageRequestCount > 1 ? "s" : ""} for${backendProcessingRequestNow ? " your request" : " "}:<br>'${queueList[0].prompt}'...</b>`;

        // Add the rest of the queued list of requests to the UI:
        let queuePosition = 1;
        let imageCount = 0;
        if (queueList.length > 1)
        {
            const orderedList = document.createElement("ol");
            for(let queueIndex = 1; queueIndex < queueList.length; queueIndex += 1)
            {
                let queueItem = queueList[queueIndex];
                const listItem = document.createElement("li");
                listItem.innerText = `${queueItem.prompt} - (${queueItem.num_images} image${queueItem.num_images > 1 ? "s" : ""})`;
                imageCount += queueItem.num_images;

                // If the queue_id matches the one returned to use by the AI, this is our request, so highlight it:
                if (queueItem.queue_id === global_currentQueueId)
                {
                    listItem.style.fontWeight = "bold";
                    listItem.style.backgroundColor = "green";
                    backendProcessingRequestNow = true;

                    // Mention this in the status message:
                    document.getElementById('status').innerText = `Request queued - position: ${queuePosition}`;
                    imageCount += queueItem.num_images;
                }
                orderedList.appendChild(listItem);
                queuePosition += 1;
            }
            queueUI.appendChild(orderedList);
        }
        else
        {
            queueUI.innerHTML += " >> Queue is Empty!"
        }
    }


}

/**
 * Retrieve the library of images in JSON format, which we will use to display the images
 * where the queue_id returned by the AI matches the queue_id of the request we are currently processing.
 * @returns {Promise<boolean|*[]|any>}
 */
const getLibrary = async () =>
{
    let rawResponse;
    document.getElementById('status').innerText = "Reading library...";

    try
    {
        rawResponse = await fetch('/getlibrary', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });
    }
    catch (e)
    {
        document.getElementById('status').innerText = "Sorry, service offline";
        return false;
    }

    if (rawResponse.status === 200)
    {
        document.getElementById('status').innerText = "Ready";
        return await rawResponse.json();
    }
    else
    {
        if (rawResponse.status === 502)
        {
            document.getElementById('status').innerText = `AI currently powering up and will start work on queued requests soon.`;
            return [];
        }
        else
        {
            if (rawResponse.status === 404)
            {
                document.getElementById('status').innerText = "SD Engine Status: Online and ready";
                return [];
            }
            else
            {
                document.getElementById('status').innerText = `Sorry, an HTTP error ${rawResponse.status} occurred - check again shortly!`;
                return [];
            }
        }
    }
}


const displayCalculatedImageCount = () =>
{
    let imageCount = parseInt(document.getElementById('num_images').value);
    const maxDDIMSteps = document.getElementById("min_ddim_steps") ? parseInt(document.getElementById("min_ddim_steps").value) : 0;
    const minDDIMSteps = document.getElementById("max_ddim_steps") ? parseInt(document.getElementById("max_ddim_steps").value) : 0;
    imageCount = imageCount * (maxDDIMSteps - minDDIMSteps + 1);
    document.getElementById('estimated_time').innerHTML = `<i>${imageCount} image${imageCount > 1 ? "s" : ""} to be created</i>`;
}

/**
 * Loop through the library looking for our queue_id and return/display the actual images.
 * @param imageList
 */
const displayImages = (imageList) =>
{
    const timestamp = new Date().getTime();    // used to force a reload of the image and not get a cached copy
    const masterImage = document.getElementById("master_image");

    masterImage.src = imageList.length > 0 ? `${imageList[imageList.length - 1]}?timestamp=${timestamp}` : "/blank.png";

    // Now the master image had been updated, we can display the rest of the images in their correct aspect ratios:
    const widthHeightRatio = masterImage.height / masterImage.width;

    for(let imageIndex = 0; imageIndex < imageList.length; imageIndex += 1)
    {
        const image = document.getElementById(`image_${imageIndex}`);
        image.src = imageList[imageIndex] ? `${imageList[imageIndex]}?timestamp=${timestamp}` : "/blank.png";
        image.width = image.height / widthHeightRatio;
    }
}

/**
 * Creates the image placeholders in the UI which will be populated with actual images as processing progresses
 */
const createImagePlaceHolders = () =>
{
    const output = document.getElementById("output");
    output.innerHTML = ""; //Empty of all child HTML ready for new images to be added (it should be empty anyway).

    const masterImage = document.createElement("img");
    masterImage.id = `master_image`;
    masterImage.style.zIndex = "0";

    // Update the master_image with teh most recent image in the list
    masterImage.src = "/blank.png";
    output.appendChild(masterImage);
    const p = document.createElement("p")
    p.id = "master_image_caption";
    output.appendChild(p);

    const includesOriginalImage =  document.getElementById("original_image_path") && document.getElementById("original_image_path").value !== "";

    let imageElementsToCreate = includesOriginalImage ? global_imagesRequested + 1 : global_imagesRequested;

    // multiply the number of images required by the number of the difference in ddim_steps
    const maxDDIMSteps = document.getElementById("min_ddim_steps") ? parseInt(document.getElementById("max_ddim_steps").value) : 0;
    const minDDIMSteps = document.getElementById("min_ddim_steps") ? parseInt(document.getElementById("max_ddim_steps").value) : 0;
    imageElementsToCreate = imageElementsToCreate * (maxDDIMSteps - minDDIMSteps + 1);


    for(let imageIndex = 0; imageIndex < imageElementsToCreate; imageIndex += 1)
    {
        const image = document.createElement("img");
        image.id = `image_${imageIndex}`;
        image.src = "/blank.png";
        image.height = 150;
        image.width = 150;
        image.style.zIndex = "0";
        image.style.position = "relative";

        image.onmouseover = function ()
        {
            this.style.transform = "scale(1.5)";
            this.style.transform += `translate(0px,0px)`;
            this.style.transition = "transform 0.25s ease";
            this.style.zIndex = "100";
            const masterImage = document.getElementById(`master_image`);
            const masterImageCaption = document.getElementById(`master_image_caption`);
            masterImage.src = this.src;
            const srcElements = this.src.split("/");
            const imageNameSections = srcElements[5].split("-");
            masterImageCaption.innerText = `Image ${imageNameSections[0]}, DDIM step ${imageNameSections[1]}`;
        };
        image.onmouseleave = function ()
        {
            this.style.transform = "scale(1)";
            this.style.transform += "translate(0px,0px)";
            this.style.transition = "transform 0.25s ease";
            this.style.zIndex = "0";
        };
        output.appendChild(image);
    }
}



function estimateCountdownTimeSeconds(imageCount)
{
    let countdownValue = imageCount * SECS_PER_IMAGE;

    if (document.body.innerText.includes("DDIM Steps:"))
    {
        countdownValue = global_countdownValue * parseInt(document.getElementById('ddim_steps').value) / 50;
    }
    if (document.body.innerText.includes("Height:"))
    {
        const height = parseInt(document.getElementById('height').value);
        countdownValue = countdownValue * Math.pow(height / 512, 2);
    }
    if (document.body.innerText.includes("Width:"))
    {
        const width = parseInt(document.getElementById('width').value);
        countdownValue = countdownValue * Math.pow(width / 512, 2);
    }

    countdownValue = Math.floor(countdownValue);
    return countdownValue;
}

/**
 * Start the countdown timer to indicate when our images should be ready
 * @param requestedImageCount
 * @param queuePosition
 * @returns {Promise<void>}
 */
const startCountDown = async (requestedImageCount) =>
{
    // Have a guess about how long this is going to take
    console.log("Starting countdown - global_countdownTimerIntervalId = " + global_countdownTimerIntervalId);
    const status = document.getElementById("status");
    let countdownSeconds = 0;
    // Measure the time taken between each image becoming available
    let previousImageCount = 0;
    let previousImageTime = new Date().getTime();
    const includesOriginalImage =  document.getElementById("original_image_path") && document.getElementById("original_image_path").value !== "";

    // set up the countdown interval function
    global_countdownTimerIntervalId = setInterval(async () =>
    {
        // Find out how many images have already been created for this queue_id (set in global_currentQueueId))
        const currentImageListData = await getImageList();
        const currentImageCount = includesOriginalImage ? currentImageListData['images'].length - 1 : currentImageListData['images'].length;

        if (currentImageCount > previousImageCount)
        {
            // If the number of images has increased, then we can recalculate the estimated time
            const currentImageTime = new Date().getTime();
            const secsPerImage = Math.ceil((currentImageTime - previousImageTime) / 1000);
            countdownSeconds = secsPerImage * (requestedImageCount - currentImageCount);
            previousImageCount = currentImageCount;
            previousImageTime = currentImageTime;
            await displayImages(currentImageListData['images']);
        }
        else if (currentImageCount === requestedImageCount || currentImageListData['completed'])
        {
            console.log("All images are ready - stopping countdown - global_countdownTimerIntervalId = " + global_countdownTimerIntervalId);
            clearInterval(global_countdownTimerIntervalId);
            global_countdownTimerIntervalId = null;
            document.getElementById("status").innerText = "Processing completed";
            if(currentImageCount < requestedImageCount)
            {
                document.getElementById("status").innerText += " - some DDIM steps failed to process";
            }
            document.getElementById("buttonGo").innerText = "Click to send request";
            document.getElementById("buttonGo").enabled = true;

            // This sleep() pause ensures that the backend has finished writing the images before we try to download
            // them for the final time. If we don't do this, then the download may cause partial images to be downloaded.
            // If partial images were previously downloaded then this final sleep will correct any issues.
            await sleep(1);
            await displayImages(currentImageListData['images']);
        }
        else
        {
            status.innerHTML = `<i>${currentImageCount < 0 ? 0 : currentImageCount} of ${requestedImageCount} images created so far - results available `;

            if (countdownSeconds > 0)
            {
                status.innerHTML += `in about <b>${countdownSeconds}</b> second${countdownSeconds === 1 ? '' : 's'}...</i>`;
            }
            else
            {
                status.innerHTML += `shortly...</i>`;
            }
        }
    }, 1000); // the countdown will trigger every 1 second


}

const ensureDDIMStepsAreValid = (ddim_control) => {
    const originalImagePath = document.getElementById("original_image_path").value;
    let maxDDIMSteps = parseInt(document.getElementById("max_ddim_steps").value);
    let minDDIMSteps = parseInt(document.getElementById("min_ddim_steps").value);
    document.getElementById(ddim_control.id + '_value').innerText = ddim_control.value;

    const lockDDIMControls = originalImagePath !== "";

    if(ddim_control.id === "min_ddim_steps")
    {
        if(lockDDIMControls || minDDIMSteps > maxDDIMSteps)
        {
            document.getElementById("max_ddim_steps").value = minDDIMSteps;
            maxDDIMSteps = minDDIMSteps;
        }
    }
    else
    {
        if(lockDDIMControls || maxDDIMSteps < minDDIMSteps)
        {
            document.getElementById("min_ddim_steps").value = maxDDIMSteps;
            minDDIMSteps = maxDDIMSteps;
        }
    }

    document.getElementById("min_ddim_steps_value").innerText = minDDIMSteps.toString();
    document.getElementById("max_ddim_steps_value").innerText = maxDDIMSteps.toString();

    displayCalculatedImageCount();
}


const populateControlsFromHref = () =>
{
    if (window.location.href.includes("?"))
    {
        const params = new URLSearchParams(window.location.search);
        if (params.has('prompt'))
        {
            document.getElementById('prompt').value = params.get('prompt');
        }
        if (params.has('original_image_path'))
        {
            document.getElementById('original_image_path').value = params.get('original_image_path');
        }
        if (params.has('strength'))
        {
            document.getElementById('strength').value = params.get('strength');
            document.getElementById('strength_value').innerText = params.get('strength');
        }
        if (params.has('seed'))
        {
            document.getElementById('seed').value = params.get('seed');
            document.getElementById('seed_value').innerText = params.get('seed');
        }
        if (params.has('width'))
        {
            document.getElementById('width').value = params.get('width');
            document.getElementById('width_value').innerText = params.get('width');
        }
        if (params.has('height'))
        {
            document.getElementById('height').value = params.get('height');
            document.getElementById('height_value').innerText = params.get('height');
        }
        if (params.has('ddim_steps'))
        {
            document.getElementById('max_ddim_steps').value = params.get('ddim_steps');
            document.getElementById('max_ddim_steps_value').innerText = params.get('ddim_steps');
            document.getElementById('min_ddim_steps').value = params.get('ddim_steps');
            document.getElementById('min_ddim_steps_value').innerText = params.get('ddim_steps');
        }
        if (params.has('min_ddim_steps'))
        {
            document.getElementById('min_ddim_steps').value = params.get('min_ddim_steps');
            document.getElementById('min_ddim_steps_value').innerText = params.get('min_ddim_steps');
        }
        if (params.has('max_ddim_steps'))
        {
            document.getElementById('max_ddim_steps').value = params.get('max_ddim_steps');
            document.getElementById('max_ddim_steps_value').innerText = params.get('max_ddim_steps');
        }

        if (params.has('scale'))
        {
            document.getElementById('scale').value = params.get('scale');
            document.getElementById('scale_value').innerText = params.get('scale');
        }
        if (params.has('downsampling_factor'))
        {
            if (params.get('downsampling_factor') === "2")
            {
                document.getElementById('ds2').checked = true;
            }
            else if (params.get('downsampling_factor') === "4")
            {
                document.getElementById('ds4').checked = true;
            }
            else if (params.get('downsampling_factor') === "8")
            {
                document.getElementById('ds8').checked = true;
            }
            else if (params.get('downsampling_factor') === "16")
            {
                document.getElementById('ds16').checked = true;
            }
        }
    }
}

const toggleDarkMode = () =>
{
    let element = document.body;
    element.classList.toggle('dark-mode');
    localStorage.setItem("dark-mode", element.classList.contains('dark-mode') ? "Y" : "N");
}


const setDarkModeFromLocalStorage = () =>
{
    if (localStorage.getItem("dark-mode") === "Y")
    {
        document.body.classList.add('dark-mode');
    }
}



/**
 * A useful sleep function
 * A calling function can use 'await sleep(1);' to pause for 1 second
 * @param seconds
 * @returns {Promise<unknown>}
 */
const sleep = (seconds) =>
{
    return new Promise(resolve => setTimeout(resolve, seconds * 1000));
}
/**
 * Set a timer to go and get the queued prompt requests from the server every 2 seconds
 * NB: Ths does not put a strain on the (python) web server as turnaround is only 10-20 milliseconds
 * so evn if a lot of people are using the service simultaneously it easily copes (I used apache AB to test!)
 */

/**
 * Retrieve the queue and display it - set by a 2-second interval timer
 * @returns {Promise<void>}
 */
const retrieveAndDisplayCurrentQueue = async () =>
{
    const queueResponse = await fetch('/queue_status', {
        method: 'GET',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    });

    if (queueResponse.status === 200)
    {
        // display the queue
        const queueData = await queueResponse.json();
        await displayQueue(queueData);

        if(queueData.length > 0)
        {
            // if our queue_is is found at the top of the queue, the backend is processing our request so we
            // need to start the countdown timer (unless it's already running)
            if (queueData[0].queue_id === global_currentQueueId && !global_countdownTimerIntervalId)
            {
                const maxDDIMSteps = queueData[0].max_ddim_steps ? queueData[0].max_ddim_steps : 0;
                const minDDIMSteps = queueData[0].min_ddim_steps ? queueData[0].min_ddim_steps : 0;
                await startCountDown(queueData[0].num_images * (maxDDIMSteps - minDDIMSteps + 1));
            }
        }
    }
}
setInterval(retrieveAndDisplayCurrentQueue, 2000);
