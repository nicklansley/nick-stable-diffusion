const SECS_PER_IMAGE = 4; // depends on GPU image creation speed - 4 works well for me
let global_currentQueueId = '';
let global_imagesRequested = 0;
let global_countdownValue = 0;
let global_countdownTimerIntervalId = null;
let global_imageLoading = false;
let defaultUpscaleFactor = 4;


/**
 * This function is called when the user clicks the "Go" button
 * @param format  can be 'video' or 'image'
 * @returns {Promise<void>}
 */
const go = async (format) =>
{
    if (document.getElementById('prompt').value.trim() === "")
    {
        document.getElementById('status').innerText = "I need a prompt, please!"
    }
    else
    {
        document.getElementById('status').innerText = `Creating ${format}s...`
        document.getElementById('buttonGo').innerText = `Creating ${format}s...`;
        document.getElementById('buttonGo').enabled = false;

        const imagesRequested = format === 'video' ? 'num_video_frames' : 'num_images';
        global_imagesRequested = parseInt(document.getElementById(imagesRequested).value);

        //If the timer is already running and the button is clicked again, the user is sending a new request
        //to be added to the queue. So we need to stop the countdown timer, as the countdown is no longer
        //applicable to this new current request.
        if (global_countdownTimerIntervalId)
        {
            clearInterval(global_countdownTimerIntervalId);
            global_countdownTimerIntervalId = null;
        }

        const data = prepareRequestData();
        data['format'] = format;
        const rawResponse = await sendPromptRequest(data);
        await processPromptRequestResponse(rawResponse);
    }
}


const prepareRequestData = () =>
{
    const data = {
        prompt: document.getElementById("prompt").value,
        num_images: global_imagesRequested,
        seed: 0
    }

    //We have the advanced options incoming for the request from advanced.html
    if (document.getElementById("seed"))
    {
        data['seed'] = parseInt(document.getElementById("seed").value);
        if (data['seed'] === '')
        {
            data['seed'] = 0;
        }
    }

    if (document.getElementById("num_video_frames"))
    {
        data['num_video_frames'] = parseInt(document.getElementById("num_video_frames").value);
    }

    if (document.getElementById("frames_per_second"))
    {
        data['frames_per_second'] = parseInt(document.getElementById("frames_per_second").value);
    }

    if (document.getElementById("height"))
    {
        data['height'] = parseInt(document.getElementById("height").value);
        if (data['height'] === '')
        {
            data['height'] = 512;
        }
    }

    if (document.getElementById("width"))
    {
        data['width'] = parseInt(document.getElementById("width").value);
        if (data['width'] === '')
        {
            data['width'] = 512;
        }
    }

    if (document.getElementById("ddim_steps"))
    {
        data['max_ddim_steps'] = parseInt(document.getElementById("ddim_steps").value);
        data['min_ddim_steps'] = data['max_ddim_steps']
    }
    else
    {
        if (document.getElementById("min_ddim_steps"))
        {
            data['min_ddim_steps'] = parseInt(document.getElementById("min_ddim_steps").value);
            data['max_ddim_steps'] = parseInt(document.getElementById("max_ddim_steps").value);
        }
    }

    if (document.getElementById("scale"))
    {
        data['scale'] = parseFloat(document.getElementById("scale").value);
        if (data['scale'] === '')
        {
            data['scale'] = 7.5;
        }
    }

    if (document.getElementById("auto_upscale"))
    {
        data['auto_upscale'] = document.getElementById("auto_upscale") ? document.getElementById("auto_upscale").checked : false;
    }

    if (document.getElementById("original_image_path"))
    {
        data['original_image_path'] = document.getElementById("original_image_path").value;
    }

    if (document.getElementById("zoom_factor"))
    {
        data['zoom_factor'] = parseFloat(document.getElementById("zoom_factor").value);
        if (data['zoom_factor'] === '')
        {
            data['zoom_factor'] = 1.1;
        }
    }

    // Note that the strength slider represents a value fomo 0.01 - 99.9% whereas strength is a float from 0.0 to 0.999
    // I used percentage because it is easier to understand for the user.
    if (document.getElementById("strength"))
    {
        data['strength'] = parseFloat(document.getElementById("strength").value) / 100;
        if (data['strength'] === '')
        {
            data['strength'] = 0.75;
        }
    }
    if (document.getElementById("downsampling_factor"))
    {
        data['downsampling_factor'] = getDownSamplingFactor();
    }

    return data;
}


const getDownSamplingFactor = () =>
{
    let downsamplingFactor = 8;
    try
    {
        const dsfRadioGroup = document.getElementsByName("downsampling_factor");
        for(let i = 0; i < dsfRadioGroup.length; i++)
        {
            if (dsfRadioGroup[i].checked)
            {
                downsamplingFactor = dsfRadioGroup[i].value;
                break;
            }
        }
    }
    catch (e)
    {
        downsamplingFactor = 8;
    }
    return downsamplingFactor;
}

const sendPromptRequest = async (data) =>
{
    document.getElementById("output").innerText = "";

    const rawResponse = await fetch(`/${data['format']}`, {
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
    const imageListResponse = await fetch('/image_list', {
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
        return {'completed': true, 'images': []};  // error condition
    }


}

/**
 * Creates an OrderList HTML element with all the items for the queue in it
 * @param queueList
 * @param imageCount
 * @param queueUI
 * @param backendProcessingRequestNow
 * @returns {HTMLOListElement}
 */
function createQueueOrderedList(queueList, imageCount, queueUI, backendProcessingRequestNow)
{
    const orderedList = document.createElement("ol");
    for(let queueIndex = 1; queueIndex < queueList.length; queueIndex += 1)
    {
        let queueItem = queueList[queueIndex];
        const listItem = document.createElement("li");
        if (queueItem['command'] === "prompt")
        {
            listItem.innerText = `Image prompt request with prompt '${queueItem.prompt}' (${queueItem.num_images} image${queueItem.num_images > 1 ? "s" : ""})`;
            imageCount += queueItem.num_images;
        }
        else if (queueItem['command'] === "upscale")
        {
            listItem.innerText = `Upscale/Enhance request for ${queueItem['image_list'].length} image${queueItem['image_list'].length > 1 ? "s" : ""}`;
        }
        else if (queueItem['command'] === 'video')
        {
            listItem.innerText = `Create video request with prompt '${queueItem['prompt']}' (${queueItem['num_video_frames']} frames)`;
        }
        // If the queue_id matches the one returned to use by the AI, this is our request, so highlight it:
        if (queueItem.queue_id === global_currentQueueId)
        {
            listItem.style.fontWeight = "bold";
            listItem.style.backgroundColor = "green";
            backendProcessingRequestNow = true;

            // Mention this in the status message:
            document.getElementById('status').innerText = `Request queued - position: ${queueIndex + 1}`;
            imageCount += queueItem.num_images;
        }
        orderedList.appendChild(listItem);
    }
    return orderedList;
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
        const topQueueItem = queueList[0];
        // Is my request being currently processed? If it's first in the queue, then yes.
        backendProcessingRequestNow = queueList[0].queue_id === global_currentQueueId

        const maxDDIMSteps = topQueueItem.max_ddim_steps ? topQueueItem.max_ddim_steps : 0;
        const minDDIMSteps = topQueueItem.min_ddim_steps ? topQueueItem.min_ddim_steps : 0;

        const imageRequestCount = topQueueItem.num_images * (maxDDIMSteps - minDDIMSteps + 1);

        // The first item in the queue is the one that the AI is currently processing, so display the info
        // separately from the queue list
        const processingDiv = document.createElement("div");
        let showText = '';

        // Text varies whether it is a prompt, video or upscale command
        if (topQueueItem['command'] === 'prompt')
        {
            showText = `<b>Now creating ${imageRequestCount} image${imageRequestCount > 1 ? "s" : ""} for${backendProcessingRequestNow ? " your request" : " "}:<br>'${topQueueItem.prompt}'...</b>`;
        }
        else if (topQueueItem['command'] === 'upscale')
        {
            showText = `<b>Upscale/Enhance request for ${topQueueItem['image_list'].length} image${topQueueItem['image_list'].length > 1 ? "s" : ""}</b>`;
        }
        else if (topQueueItem['command'] === 'video')
        {
            showText = `<b>Creating video with prompt '${topQueueItem['prompt']}' (${topQueueItem['num_video_frames']} frames)</b>`;
        }

        queueUI.innerHTML = `${showText}<br>Current queue:<br>`;
        processingDiv.innerHTML = showText;

        // Now display the rest of the queue list:
        let imageCount = 0;
        if (queueList.length > 1)
        {
            // Create an OrderList HTML element with all the items for the queue in it, and append it to the queueUI section@
            queueUI.appendChild(createQueueOrderedList(queueList, imageCount, queueUI, backendProcessingRequestNow));
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


const validateImageCountInput = () =>
{
    let imageCount = parseInt(document.getElementById('num_images').value);
    const minDDIMSteps = document.getElementById("min_ddim_steps") ? parseInt(document.getElementById("min_ddim_steps").value) : 0;
    const maxDDIMSteps = document.getElementById("max_ddim_steps") ? parseInt(document.getElementById("max_ddim_steps").value) : 0;
    if (minDDIMSteps !== maxDDIMSteps)
    {
        document.getElementById('num_images').value = 1;
        document.getElementById('num_images_value').innerText = `1 image, because the DDIM Steps setting (from ${minDDIMSteps} to ${maxDDIMSteps}) will generate ${maxDDIMSteps - minDDIMSteps + 1} images`;
    }
    else
    {
        document.getElementById('num_images_value').innerText = `${imageCount} image${imageCount > 1 ? "s" : ""}`;
    }
}


const authorDescriptionFromImageFileName = (imageFileName) =>
{
    if (imageFileName.includes("blank."))
    {
        return '';
    }
    if (imageFileName.includes("original."))
    {
        return 'Original input image'
    }
    if (imageFileName.includes("_upscaled."))
    {
        return 'Upscaled version - click to view full size in a new browser window';
    }

    const srcElements = imageFileName.split("/");
    const imageNameSections = srcElements[5].split("-");
    const imageNumber = imageNameSections[0]
    const ddimSteps = imageNameSections[1].replace('D', '')
    const scale = imageNameSections[2].replace('S', '');
    const seedValue = imageNameSections[3].replace('R', '');
    return `New Image - click to edit in Advanced page - #${imageNumber}, DDIM steps: ${ddimSteps}, Scale: ${scale}, Seed: ${seedValue}`;
}

/**
 * Loop through the library looking for our queue_id and return/display the actual images.
 * @param imageList
 */
const displayImages = async (imageList) =>
{
    const masterImage = document.getElementById("master_image");
    const masterImageCaption = document.getElementById("master_image_caption");

    masterImage.src = imageList.length > 0 ? imageList[imageList.length - 1] : "/blank.png";
    if (!masterImage.src.includes("blank.") && !masterImage.src.includes("original."))
    {
        masterImageCaption.innerText = authorDescriptionFromImageFileName(masterImage.src)
    }

    // Now the master image had been updated, we can display the rest of the images in their correct aspect ratios:
    for(let imageIndex = 0; imageIndex < imageList.length; imageIndex += 1)
    {
        const image = document.getElementById(`image_${imageIndex}`);
        image.src = imageList[imageIndex] ? imageList[imageIndex] : "/blank.png";
        image.style.height = "150px";
        image.style.width = "auto";
        global_imageLoading = true;
        while (global_imageLoading)
        {
            // We wait for this image's onload event to complete and set global_imageLoading to false before moving on to the next image:
            await sleep(100);
        }
        if (image.src.includes('_upscaled.'))
        {
            image.style.borderColor = "gold"; // Upscaled image -
            image.style.borderWidth = "5px";
        }
    }
}


const displayVideoFrame = async (imageList) =>
{
    // We only display the most recent frame, re-used as '00-original' when creating the next frame.
    const masterImage = document.getElementById("master_image");
    masterImage.src = imageList[imageList.length - 1];
}

const displayFinishedVideo = async (imageList) =>
{
    // create a video element
    const videoElement = document.createElement('video');
    videoElement.controls = true;
    videoElement.autoplay = true;
    videoElement.loop = true;


    // The video is always called video.mp4 so take the 'original' entry and replace it with 'video.mp4'
    // I am sure there are quicker ways to extract the library path to the video and image  files than this but hey!
    for(const imagePath of imageList)
    {
        if (imagePath.includes('original.jpg'))
        {
            videoElement.src = imagePath.replace('00-original.jpg', 'video.mp4');
            break;
        }
        else if (imagePath.includes('original.png'))
        {
            videoElement.src = imagePath.replace('00-original.png', 'video.mp4');
            break;
        }
    }

    //replace the document element 'master_image' with the video element
    document.getElementById('output').innerHTML = "";
    document.getElementById('output').appendChild(videoElement);
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
    masterImage.style.height = "512px"
    masterImage.style.width = "auto";
    masterImage.style.zIndex = "0";

    // Update the master_image with teh most recent image in the list
    masterImage.src = "/blank.png";
    output.appendChild(masterImage);
    const p = document.createElement("p")
    p.id = "master_image_caption";
    output.appendChild(p);

    const includesOriginalImage = document.getElementById("original_image_path") && document.getElementById("original_image_path").value !== "";


    // multiply the number of images required by the number of the difference in ddim_steps
    const maxDDIMSteps = document.getElementById("max_ddim_steps") ? parseInt(document.getElementById("max_ddim_steps").value) : 0;
    const minDDIMSteps = document.getElementById("min_ddim_steps") ? parseInt(document.getElementById("min_ddim_steps").value) : 0;
    let imageElementsToCreate = global_imagesRequested + (maxDDIMSteps - minDDIMSteps);
    const upscaleFlag = document.getElementById("auto_upscale") ? document.getElementById("auto_upscale").checked : false;

    if (document.getElementById("frames_per_second"))
    {
        // This is the video page
        imageElementsToCreate = 1;
    }
    else if (upscaleFlag)
    {
        imageElementsToCreate = (imageElementsToCreate * 2);
    }
    imageElementsToCreate = includesOriginalImage ? imageElementsToCreate + (upscaleFlag ? 2 : 1) : imageElementsToCreate;


    for(let imageIndex = 0; imageIndex < imageElementsToCreate; imageIndex += 1)
    {
        const image = document.createElement("img");
        image.id = `image_${imageIndex}`;
        image.src = "/blank.png";
        image.height = 150;
        image.width = 150;
        image.style.zIndex = "0";
        image.style.position = "relative";

        image.onload = () =>
        {
            global_imageLoading = false;
        }

        image.onclick = function ()
        {
            if (image.src.includes('_upscaled.'))
            {
                // This is an upscaled image so clicking it will open it in a new browser window
                window.open(image.src, '_blank');
            }
            else
            {
                const libraryItem = {
                    text_prompt: document.getElementById("prompt").value,
                    seed: getSeedValueFromImageFileName(this.src),
                    height: document.getElementById("height") ? document.getElementById("height").value : 512,
                    width: document.getElementById("width") ? document.getElementById("width").value : 512,
                    min_ddim_steps: document.getElementById("min_ddim_steps") ? document.getElementById("min_ddim_steps").value : 40,
                    max_ddim_steps: document.getElementById("max_ddim_steps") ? document.getElementById("max_ddim_steps").value : 40,
                    ddim_eta: 0,  // not used by the UI but available to the API
                    scale: document.getElementById("scale") ? document.getElementById("scale").value : 7.5,
                    downsampling_factor: getDownSamplingFactor()
                }
                const urlencoded_image_src = this.src;
                window.open(`${createLinkToAdvancedPage(urlencoded_image_src, libraryItem)}`, '_self');
            }
        }

        image.onmouseover = function ()
        {
            this.style.transform = "scale(1.5)";
            this.style.transform += `translate(0px,0px)`;
            this.style.transition = "transform 0.25s ease";
            this.style.zIndex = "100";
            const masterImage = document.getElementById(`master_image`);
            const masterImageCaption = document.getElementById(`master_image_caption`);
            masterImage.src = this.src;
            masterImageCaption.innerText = authorDescriptionFromImageFileName(this.src)
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
 * @param imageOrVideoFrameText
 * @returns {Promise<void>}
 */
const startCountDown = async (requestedImageCount, imageOrVideoFrameText) =>
{
    // Have a guess about how long this is going to take
    console.log("Starting countdown - global_countdownTimerIntervalId = " + global_countdownTimerIntervalId);
    const status = document.getElementById("status");
    // Measure the time taken between each image becoming available
    let previousImageCount = 0;
    const includesOriginalImage = document.getElementById("original_image_path") && document.getElementById("original_image_path").value !== "";

    // set up the countdown interval function
    global_countdownTimerIntervalId = setInterval(async () =>
    {
        // Find out how many images have already been created for this queue_id (set in global_currentQueueId))
        const currentImageListData = await getImageList();
        const currentImageCount = includesOriginalImage ? currentImageListData['images'].length - 1 : currentImageListData['images'].length;

        if (currentImageCount > previousImageCount)
        {
            // If the number of images has increased, then we can recalculate the estimated time
            previousImageCount = currentImageCount;
            if (document.getElementById("frames_per_second"))
            {
                await displayVideoFrame(currentImageListData['images']);
            }
            else
            {
                await displayImages(currentImageListData['images']);
            }
        }
        else if (currentImageCount === requestedImageCount || currentImageListData['completed'])
        {
            console.log("All images are ready - stopping countdown - global_countdownTimerIntervalId = " + global_countdownTimerIntervalId);
            clearInterval(global_countdownTimerIntervalId);
            global_countdownTimerIntervalId = null;
            if (document.getElementById("frames_per_second"))
            {
                document.getElementById("status").innerText = "Processing completed! If video does not load on this page, you will find it in the Library";
            }
            else
            {
                document.getElementById("status").innerText = "Processing completed! If not all images appear on this page, you will find them all in the Library";
            }
            if (document.getElementById("auto_upscale") ? document.getElementById("auto_upscale").checked : false)
            {
                document.getElementById("status").innerText += " - upscaled versions will appear automatically with 'gold' border";
            }

            if (currentImageCount < requestedImageCount)
            {
                document.getElementById("status").innerText += " - some DDIM steps failed to process";
            }
            document.getElementById("buttonGo").innerText = "Click to send request";
            document.getElementById("buttonGo").enabled = true;

            // This sleep() pause ensures that the backend has finished writing the images before we try to download
            // them for the final time. If we don't do this, then the download may cause partial images to be downloaded.
            // If partial images were previously downloaded then this final sleep will correct any issues.
            await sleep(1000);
            if (document.getElementById("frames_per_second"))
            {
                await displayFinishedVideo(currentImageListData['images']);
            }
            else
            {
                await displayImages(currentImageListData['images']);
            }
        }
        else
        {
            status.innerHTML = `<i>${currentImageCount < 0 ? 0 : currentImageCount} of ${requestedImageCount} ${imageOrVideoFrameText} created so far`;
        }
    }, 1000); // the countdown will trigger every 1 second
}


const ensureDDIMStepsAreValid = (ddim_control) =>
{
    let maxDDIMSteps = parseInt(document.getElementById("max_ddim_steps").value);
    let minDDIMSteps = parseInt(document.getElementById("min_ddim_steps").value);
    document.getElementById(ddim_control.id + '_value').innerText = ddim_control.value;


    if (ddim_control.id === "min_ddim_steps")
    {
        if (minDDIMSteps > maxDDIMSteps)
        {
            document.getElementById("max_ddim_steps").value = minDDIMSteps;
            maxDDIMSteps = minDDIMSteps;
        }
    }
    else
    {
        if (maxDDIMSteps < minDDIMSteps)
        {
            document.getElementById("min_ddim_steps").value = maxDDIMSteps;
            minDDIMSteps = maxDDIMSteps;
        }
    }

    document.getElementById("min_ddim_steps_value").innerText = minDDIMSteps.toString();
    document.getElementById("max_ddim_steps_value").innerText = maxDDIMSteps.toString();
    if (minDDIMSteps !== maxDDIMSteps && parseInt(document.getElementById("num_images").value) > 1)
    {
        document.getElementById("num_images").value = 1;
        document.getElementById("num_images_value").innerText = "1";
        document.getElementById("status").innerText = "Number of images set to 1 because DDIM steps are not equal";
    }
    validateImageCountInput();
}

const buttonClearImage_Clicked = (button) =>
{
    document.getElementById("original_image_path").value = "";
    document.getElementById('image_drop_area').innerHTML = '';
    button.style.visibility = "hidden";
    document.getElementById("status").innerText = "Original image cleared";
    validateImageCountInput();
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
            const dragDropImage = document.getElementById('drag_drop_image');
            dragDropImage.src = params.get('original_image_path');

            document.getElementById('button_remove_image').style.visibility = 'visible';

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
        if (params.has('ddim_steps')) // legacy from an 'old' version of the library
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


const setupImageDragDrop = () =>
{
    const imageDropArea = document.getElementById('image_drop_area');
    imageDropArea.addEventListener('dragover', (e) =>
    {
        e.preventDefault();
        e.stopPropagation();
        imageDropArea.classList.add('drag-over');
    });

    imageDropArea.addEventListener('dragleave', (e) =>
    {
        e.preventDefault();
        e.stopPropagation();
        imageDropArea.classList.remove('drag-over');
    });

    imageDropArea.addEventListener('drop', (e) =>
    {
        e.preventDefault();
        e.stopPropagation();
        imageDropArea.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file)
        {
            const reader = new FileReader();
            reader.onload = (e) =>
            {
                const img = new Image();
                img.onload = () =>
                {
                    document.getElementById('original_image_path').value = e.target.result;
                    const dragDropImage = document.getElementById('drag_drop_image');
                    dragDropImage.src = e.target.result;
                    if (dragDropImage.height > 300)
                    {
                        dragDropImage.style.height = "300px";
                        dragDropImage.style.width = "auto";
                    }
                    document.getElementById('original_image_path').dispatchEvent(new Event('change'));
                }
                img.src = e.target.result;
            }
            reader.readAsDataURL(file);
            document.getElementById('button_remove_image').style.visibility = 'visible';
        }
    });
}

const createLinkToAdvancedPage = (image_src, libraryItem) =>
{
    // Remove the domain name before '/library'
    const adjustedImageSrc = image_src.replace(image_src.substring(0, image_src.indexOf('/library') + 1), '');
    const urlencoded_image_src = encodeURIComponent(adjustedImageSrc);
    const urlEncodedPrompt = encodeURIComponent(libraryItem['text_prompt']);
    let seedValue = getSeedValueFromImageFileName(image_src);
    if (seedValue === '')
    {
        seedValue = libraryItem['seed'];
    }
    return `advanced.html?original_image_path=${urlencoded_image_src}&prompt=${urlEncodedPrompt}&seed=${seedValue}&height=${libraryItem['height']}&width=${libraryItem['width']}&min_ddim_steps=${libraryItem['min_ddim_steps']}&max_ddim_steps=${libraryItem['max_ddim_steps']}&ddim_eta=${libraryItem['ddim_eta']}&scale=${libraryItem['scale']}&downsampling_factor=${libraryItem['downsampling_factor']}`;
}

const getSeedValueFromImageFileName = (imageFileName) =>
{
    if (imageFileName.includes("blank."))
    {
        return '';
    }
    if (imageFileName.includes("original."))
    {
        return '';
    }
    const srcElements = imageFileName.split("/");
    const imageNameSections = srcElements[5].split("-");
    return imageNameSections[3].replace('R', '');
}


/**
 * A useful sleep function
 * A calling function can use 'await sleep(1);' to pause for 1 second
 * @param milliSeconds
 * @returns {Promise<unknown>}
 */
const sleep = (milliSeconds) =>
{
    return new Promise(resolve => setTimeout(resolve, milliSeconds));
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

        if (queueData.length > 0)
        {
            // if our queue_is is found at the top of the queue, the backend is processing our request so we
            // need to start the countdown timer (unless it's already running)
            if (queueData[0].queue_id === global_currentQueueId && !global_countdownTimerIntervalId)
            {
                const maxDDIMSteps = queueData[0].max_ddim_steps ? queueData[0].max_ddim_steps : 0;
                const minDDIMSteps = queueData[0].min_ddim_steps ? queueData[0].min_ddim_steps : 0;
                const imageCount = queueData[0].num_images ? queueData[0].num_images * (maxDDIMSteps - minDDIMSteps + 1) : queueData[0].num_video_frames;
                await startCountDown(imageCount, queueData[0].num_images ? 'images' : 'frames');
            }
        }
    }
}

setInterval(retrieveAndDisplayCurrentQueue, 4000);