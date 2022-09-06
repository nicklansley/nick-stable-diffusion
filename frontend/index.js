const SECS_PER_IMAGE = 4; // depends on GPU image creation speed - 4 works well for me
let global_currentQueueId = '';
let global_countdownTimerIntervalId = null;
let global_countdownValue = 0;
let global_countdownRunning = false;

/**
 * Send the text prompt to the AI and get a queue_id back in 'queue_id' which will be used to track the request.
 * @returns {Promise<void>}
 */
const go = async () =>
{
    document.getElementById('status').innerText = "Creating images..."
    document.getElementById('buttonGo').innerText = "Creating images...";
    document.getElementById('buttonGo').enabled = false;

    const data = {
        prompt: document.getElementById("prompt").value,
        num_images: document.getElementById("num_images").value,
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

        data['ddim_steps'] = parseInt(document.getElementById("ddim_steps").value);
        if (data['ddim_steps'] === '')
        {
            data['ddim_steps'] = 50;
        }

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

    document.getElementById("output").innerText = "";

    const rawResponse = await fetch('/prompt', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    if (rawResponse.status === 200)
    {
        const queueConfirmation = await rawResponse.json();
        global_currentQueueId = queueConfirmation.queue_id;
        document.getElementById('status').innerText = `Request queued - check the queue for position`;
    }
    else
    {
        document.getElementById('status').innerText = `Stable Diffusion Engine Status: Sorry, an HTTP error ${rawResponse.status} occurred - have another go!`;
    }
    document.getElementById('buttonGo').innerText = "Click to send request";
    document.getElementById('buttonGo').enabled = true;
}


/**
 * Retrieve the queue and display it.
 * @returns {Promise<void>}
 */
const retrieveAndDisplayCurrentQueue = async () =>
{
    const output = document.getElementById("output");
    const queueResponse = await fetch('/queue_status', {
        method: 'GET',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    });

    if (queueResponse.status === 200)
    {
        const queueData = await queueResponse.json();
        await displayQueue(queueData);

        //Look for our Queue ID in the queue:
        let foundQueueId = false;
        for(const queueItem of queueData)
        {
            if (queueItem.queue_id === global_currentQueueId)
            {
                foundQueueId = true;
                break;
            }
        }
        //If we did not find our queue_id then processing of our request must be completed.
        //So, if no images are being displayed, go get them!
        //However, do not this if the prompt has no value (i.e. when the page is first loaded)
        if (!foundQueueId
            && (output.innerText === 'Retrieving images...' || output.innerHTML === '' || output.innerText.startsWith('Results available in'))
            && document.getElementById("prompt").value.length > 0)
        {
            document.getElementById('status').innerText = `Image creation completed`;
            stopCountDown();
            const library = await getLibrary();
            if (library)
            {
                await displayImages(library, output);
            }
        }
    }


}


/**
 * Display the queue.
 * @param queueList
 * @returns {Promise<void>}
 */
const displayQueue = async (queueList) =>
{
    let myQueueIdIsCurrentlyBeingProcessedFlag = false;

    const queueUI = document.getElementById("queue");
    if (queueList.length === 0)
    {
        queueUI.innerHTML = "Current queue: Empty<br>You'll be first if you submit a request!";
    }
    else
    {
        // Is my request being currently processed?
        myQueueIdIsCurrentlyBeingProcessedFlag = queueList[0].queue_id === global_currentQueueId


        // The first item in the queue is the one that the AI is currently processing:
        queueUI.innerHTML = `<p><b>Now creating ${queueList[0].num_images} image${queueList[0].num_images > 1 ? "s" : ""} for${myQueueIdIsCurrentlyBeingProcessedFlag ? " your request" : " "}:<br>'${queueList[0].prompt}'...</b></p><br>Current queue:<br>`;

        const processingDiv = document.createElement("div");
        processingDiv.innerHTML = `<b>Now creating ${queueList[0].num_images} image${queueList[0].num_images > 1 ? "s" : ""} for${myQueueIdIsCurrentlyBeingProcessedFlag ? " your request" : " "}:<br>'${queueList[0].prompt}'...</b>`;

        // If we are the first in the queue, our prompt is the one currently being processed by the AI
        // so highlight it:
        if (myQueueIdIsCurrentlyBeingProcessedFlag && document.getElementById("output").innerText !== "Retrieving images...")
        {
            // Mention this in the status message:
            document.getElementById('status').innerText = `Your request is being processed right now...`;
            await startCountDown(queueList[0].num_images);
        }

        // Add the rest of the queue to the UI:

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
                    listItem.style.backgroundColor = "lightgreen";
                    myQueueIdIsCurrentlyBeingProcessedFlag = true;
                    // Mention this in the status message:
                    document.getElementById('status').innerText = `Request queued - position: ${queuePosition}`;
                    imageCount += queueItem.num_images;
                    await startCountDown(imageCount);
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
                document.getElementById('status').innerText = "DALL-E Engine Status: Online and ready";
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


const calculateEstimatedTime = () =>
{
    const imageCount = parseInt(document.getElementById('num_images').value);
    let estimatedTime = imageCount * SECS_PER_IMAGE;
    if (document.body.innerText.includes("DDIM Steps:"))
    {
        estimatedTime = estimatedTime * (parseInt(document.getElementById('ddim_steps').value) / 50);
    }
    if (document.body.innerText.includes("Height:"))
    {
        const height = parseInt(document.getElementById('height').value);
        estimatedTime = estimatedTime * Math.pow(height / 512, 2);
    }
    if (document.body.innerText.includes("Width:"))
    {
        const width = parseInt(document.getElementById('width').value);
        estimatedTime = estimatedTime * Math.pow(width / 512, 2);
    }
    document.getElementById('estimated_time').innerHTML = `<i>${imageCount} image${imageCount > 1 ? "s" : ""} - estimated time: <b>${parseInt(estimatedTime)}</b> seconds</i>`;
}

/**
 * Loop through the library looking for our queue_id and return/display the actual images.
 * @param library
 * @param output
 * @returns {Promise<void>}
 */
const displayImages = async (library, output) =>
{
    output.innerHTML = ""; //Empty of all child HTML ready for new images to be added.
    for(const libraryItem of library)
    {
        if (libraryItem.queue_id === global_currentQueueId)
        {
            if (libraryItem.error)
            {
                output.innerHTML = `<p><b>Sorry, an error occurred: ${libraryItem.error}</b></p>`;
            }
            else
            {
                const masterImage = document.createElement("img");
                if (libraryItem['generated_images'].length > 0)
                {
                    if (libraryItem['generated_images'][0].endsWith('00-original.png'))
                    {
                        masterImage.src = libraryItem['generated_images'][1]; // the second image is the first generated image when using an input image
                    }
                    else
                    {
                        masterImage.src = libraryItem['generated_images'][0]; // the first image is the first generated image
                    }

                    masterImage.id = `master_image_${libraryItem['queue_id']}`;
                    masterImage.alt = libraryItem['text_prompt'];
                    masterImage.height = libraryItem['height'];
                    masterImage.width = libraryItem['width'];
                    masterImage.style.zIndex = "0";
                    output.appendChild(masterImage);
                    output.appendChild(document.createElement("br"));
                }

                let imageCount = 0;
                for(const image_entry of libraryItem['generated_images'])
                {
                    const image = document.createElement("img");
                    image.src = image_entry;
                    image.alt = libraryItem['text_prompt'];
                    image.height = 150;
                    image.width = Math.ceil(150 * (libraryItem['width'] / libraryItem['height']));
                    image.style.zIndex = "0";
                    image.style.position = "relative";

                    image.onmouseover = function ()
                    {
                        this.style.transform = "scale(1.5)";
                        this.style.transform += `translate(0px,0px)`;
                        this.style.transition = "transform 0.25s ease";
                        this.style.zIndex = "100";
                        const masterImage = document.getElementById(`master_image_${libraryItem['queue_id']}`);
                        masterImage.src = this.src;
                    };
                    image.onmouseleave = function ()
                    {
                        this.style.transform = "scale(1)";
                        this.style.transform += "translate(0px,0px)";
                        this.style.transition = "transform 0.25s ease";
                        this.style.zIndex = "0";
                    };
                    output.appendChild(image);
                    imageCount += 1;
                }
            }
        }
    }
}

/**
 * Start the countdown timer to indicate when our images should be ready
 * @param imageCount
 * @param queuePosition
 * @returns {Promise<void>}
 */
const startCountDown = async (imageCount) =>
{
    if (!global_countdownRunning)
    {
        global_countdownValue = imageCount * SECS_PER_IMAGE;

        if (document.body.innerText.includes("DDIM Steps:"))
        {
            global_countdownValue = global_countdownValue * parseInt(document.getElementById('ddim_steps').value) / 50;
        }
        if (document.body.innerText.includes("Height:"))
        {
            const height = parseInt(document.getElementById('height').value);
            global_countdownValue = global_countdownValue * Math.pow(height / 512, 2);
        }
        if (document.body.innerText.includes("Width:"))
        {
            const width = parseInt(document.getElementById('width').value);
            global_countdownValue = global_countdownValue * Math.pow(width / 512, 2);
        }

        const output = document.getElementById("output");
        global_countdownValue = parseInt(global_countdownValue);

        output.innerHTML = `<i>Results available in about <b>${global_countdownValue}</b> second${global_countdownValue === 1 ? '' : 's'}...</i>`;

        global_countdownTimerIntervalId = setInterval(() =>
        {

            if (global_countdownValue === 1)
            {
                stopCountDown();
            }
            else
            {
                global_countdownValue -= 1;
                output.innerHTML = `<i>Results available in about <b>${global_countdownValue}</b> second${global_countdownValue === 1 ? '' : 's'}...</i>`;
            }
        }, 1000); // the countdown will trigger every 1 second

        global_countdownRunning = true;
    }
}

/**
 *
 */
const stopCountDown = () =>
{
    clearInterval(global_countdownTimerIntervalId);
    document.getElementById("output").innerHTML = "Retrieving images...";
    document.getElementById("buttonGo").innerText = "Click to send request";
    document.getElementById("buttonGo").enabled = true;
    global_countdownRunning = false;
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
            document.getElementById('ddim_steps').value = params.get('ddim_steps');
            document.getElementById('ddim_steps_value').innerText = params.get('ddim_steps');
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
 * Set a timer to go and get the queued prompt requests from the server every 2 seconds
 * NB: Ths does not put a strain on the (python) web server as turnaround is only 10-20 milliseconds
 * so evn if a lot of people are using the service simultaneously it easily copes (I used apache AB to test!)
 */
setInterval(retrieveAndDisplayCurrentQueue, 2000);
