const REFRESH_INTERVAL_SECS = 2
let library = [];
let global_upscaleImageList = [];
let DEFAULT_UPSCALE_FACTOR = 4;

// Get the library index file and return the list of library entries
const listLibrary = async () =>
{
    let rawResponse;
    document.getElementById('status').innerText = "Reading library...";

    try
    {
        rawResponse = await fetch('/get_library', {
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
        return await rawResponse.json();
    }
    else
    {
        document.getElementById('status').innerText = `Sorry, an HTTP error ${rawResponse.status} occurred - check again shortly!`;
        return [];
    }
}


// Read all the queue entries and get all the images that are currently in the queue for upscaling.
const getQueueAndListUpscaleRequests = async () =>
{
    const upscaleImageRequestList = [];
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
        for(const queueItem of queueData)
        {
            if (queueItem['command'] === "upscale")
            {
                for(const image of queueItem['image_list'])
                {
                    upscaleImageRequestList.push(image);
                }
            }
        }
        return upscaleImageRequestList;
    }
}

// Format for display all the library entries currently in the library
// Only new entries and images will be added to the web page layout
// to avoid unnecessary re-rendering of the page and image.src calls to the server
// (even if the image is already in the browser cache, the browser will still
// make a request to the server to check if the image has changed)
const formatLibraryEntries = async () =>
{
    let libraryImageCount = 0;

    // retrieve the library and sort by descending 'creation_unixtime'
    const library = await listLibrary();

    const sortedLibrary = library.sort((a, b) => a['creation_unixtime'] > b['creation_unixtime'] ? -1 : 1);

    // output is where we display the authored library entries with their info headers and images
    const output = document.getElementById("output");

    const upscaleQueueImagesList = await getQueueAndListUpscaleRequests();


    for(const libraryEntryIndex in sortedLibrary)
    {
        const libraryEntry = sortedLibrary[libraryEntryIndex];
        libraryImageCount += libraryEntry['generated_images'].length;

        // Check if this library item has already been rendered:
        const divLibraryEntryId = "div_" + libraryEntry['queue_id'];
        if (document.getElementById(divLibraryEntryId))
        {
            // IN this case the library entry already displays on the page, so we just need to update the images
            // Check if there are any new images to add to the library entry - if so, rebuild the library entry
            // and replace the existing one.
            // But, don't do this id the library entry includes a vide.mp4 file or it will restart
            // before it finishes playing:
            const imageList = document.getElementById(divLibraryEntryId).getElementsByClassName("divImage");
            if (imageList.length < libraryEntry['generated_images'].length && !libraryEntry['generated_images'][0].includes('video.mp4'))
            {
                const divRebuiltLibraryEntry = createNewDivLibraryEntry(libraryEntry, upscaleQueueImagesList);
                output.replaceChild(divRebuiltLibraryEntry, document.getElementById(divLibraryEntryId));
            }
        }
        else
        {
            // In this case the library entry does not display on the page, so we need to create a new div for it
            const divLibraryEntry = createNewDivLibraryEntry(libraryEntry, upscaleQueueImagesList);
            // Now we need to work out where it must go! We want to sort by descending creation time, so we need to
            // find the first library entry that was created before this one and insert it after that one.
            console.log('Adding new library entry');
            let inserted = false;
            for(const existingLibraryEntry of output.children)
            {
                if (existingLibraryEntry.dataset['creation_unixtime'] < libraryEntry['creation_unixtime'])
                {
                    output.insertBefore(divLibraryEntry, existingLibraryEntry);
                    inserted = true;
                    break;
                }
            }
            // If we didn't find an existing library entry that was created before this one, then we need to insert
            // this one at the end of the list
            if (!inserted)
            {
                output.appendChild(divLibraryEntry);
            }
        }
    }
    filterVisibleEntries();
    const dateNow = new Date();
    document.getElementById('status').innerText = `Updated ${dateNow.toLocaleString()} - Found ${libraryImageCount} images within ${sortedLibrary.length} library entries`;
}


const filterVisibleEntries = () =>
{
    const filterText = document.getElementById('filter').value.toLowerCase();
    const output = document.getElementById("output");
    if(filterText === "")
    {
        for(const libraryEntry of output.children)
        {
            libraryEntry.style.display = "block";
        }
    }
    else
    {
        for(const libraryEntry of output.children)
        {
            if (libraryEntry.dataset['text_prompt'].toLowerCase().includes(filterText))
            {
                libraryEntry.style.display = "block";
            }
            else
            {
                libraryEntry.style.display = "none";
            }
        }
    }
}


// Author the large master image that features in each library entry
const libraryEntry_authorMasterImagePlaceHolder = (div, thisLibraryEntry) =>
{
    const masterImage = document.createElement("img");
    if (thisLibraryEntry['generated_images'].length > 0)
    {
        if (thisLibraryEntry['generated_images'][0].includes('00-original.'))
        {
            masterImage.src = thisLibraryEntry['generated_images'][1]; // the second image is the first generated image when using an input image
        }
        else
        {
            masterImage.src = thisLibraryEntry['generated_images'][0]; // the first image is the first generated image
        }

        // Master image for group
        masterImage.id = `master_image_${thisLibraryEntry['queue_id']}`;
        masterImage.alt = thisLibraryEntry['text_prompt'];
        if(thisLibraryEntry['negative_prompt'] !== '')
            masterImage.alt += ` [${thisLibraryEntry['negative_prompt']}];`;
        masterImage.height = thisLibraryEntry['height'];
        masterImage.width = thisLibraryEntry['width'];
        masterImage.style.zIndex = "0";
        div.appendChild(masterImage);

        // Caption for master image
        const masterImageCaption = document.createElement("p")
        masterImageCaption.id = `master_image_caption_${thisLibraryEntry['queue_id']}`;
        masterImageCaption.classList.add("label");
        masterImageCaption.style.float = 'inline-start';
        div.appendChild(masterImageCaption);
    }
}

// Author an image for the library entry
const libraryEntry_addSingleImage = (imageName, image_entry, libraryItem) =>
{
    const image = document.createElement("img");
    image.id = imageName.split('.')[0];
    image.src = image_entry;
    image.alt = libraryItem['text_prompt'];
    if(libraryItem['negative_prompt'] !== '')
        image.alt += ` [${libraryItem['negative_prompt']}];`;

    image.height = 150;
    image.width = Math.ceil(150 * (libraryItem['width'] / libraryItem['height']));
    image.style.zIndex = "0";
    image.style.position = "relative";

    // Add data-image-details attribute to image using the
    // libraryItem object with generated_images list deleted.
    const dataImageDetails = JSON.parse(JSON.stringify(libraryItem));
    delete dataImageDetails['generated_images'];
    dataImageDetails.path = image_entry;

    image.onclick = function ()
    {
        window.open(image.src, '_blank');
    }

    image.onmouseover = function ()
    {
        this.style.transform = `scale(1.1)`;
        this.style.transition = "transform 0.25s ease";
        this.style.zIndex = "100";
        const masterImage = document.getElementById(`master_image_${libraryItem['queue_id']}`);
        masterImage.src = this.src;
        const masterImageCaption = document.getElementById(`master_image_caption_${libraryItem['queue_id']}`);
        masterImageCaption.innerText = authorDescriptionFromImageFileName(this.src);
    };
    image.onmouseleave = function ()
    {
        this.style.transform = "scale(1)";
        this.style.transition = "transform 0.25s ease";
        this.style.zIndex = "0";
    };
    image.oncontextmenu = function (ev)
    {
        ev.preventDefault();
        deleteImage(this);
    };
    return image;
}

// Author a single image and its control buttons for a library entry
const  libraryEntry_authorImageAndControls = (imageName, image_entry, libraryItem, imageInUpscaleQueue) =>
{
    const divImageAndButtons = document.createElement("div");
    divImageAndButtons.classList.add('divImage');
    if (imageName.includes('_upscaled.'))
    {
        divImageAndButtons.style.borderColor = "gold";
        divImageAndButtons.style.borderWidth = "5px";
    }

    const image = libraryEntry_addSingleImage(imageName, image_entry, libraryItem);
    divImageAndButtons.appendChild(image);

    const imageUpscaleOrViewButton = document.createElement("button");
    imageUpscaleOrViewButton.id = `upscale_button_${libraryItem['queue_id']}_${imageName.split('.')[0]}`;
    imageUpscaleOrViewButton.className = "button-image-action"
    if (imageName.includes('_upscaled.'))
    {
        // Button will view image in new browser window
        imageUpscaleOrViewButton.innerText = "View Upscaled";
        imageUpscaleOrViewButton.onclick = () =>
        {
            window.open(image.src, '_blank');
        }
    }
    else
    {
        // Button will push an upscale request to the queue via the upscale() function
        const alreadyUpscaled = checkIfImageAlreadyUpscaled(image.src, libraryItem['generated_images']);
        imageUpscaleOrViewButton.innerText = alreadyUpscaled ? "View Original" : imageInUpscaleQueue ? "Upscaling" : "Upscale";
        imageUpscaleOrViewButton.onclick = () =>
        {
            if (imageUpscaleOrViewButton.innerText === "Upscale")
            {
                imageUpscaleOrViewButton.innerText = 'Queued';
                imageUpscaleOrViewButton.disabled = true;
                const imageRelativePath = image.src.split("/").slice(3).join("/");
                global_upscaleImageList.push(imageRelativePath);
            }
            else
            {
                window.open(image.src, '_blank');
            }
        }

    }

    // Create an Edit button
    const imageEditButton = document.createElement("button");
    imageEditButton.className = "button-image-action"
    imageEditButton.innerText = "Edit";
    imageEditButton.onclick = () => window.open(`${createLinkToAdvancedPage(image_entry, libraryItem)}`, '_blank');


    divImageAndButtons.appendChild(document.createElement('br'))
    divImageAndButtons.appendChild(imageEditButton);
    divImageAndButtons.appendChild(imageUpscaleOrViewButton);
    return divImageAndButtons;
}

// Create a div for each image in the library item
const libraryEntry_authorImageDivs = (div, thisLibraryEntry, imageCount, upscaleQueueImagesList) =>
{
    for(const image_entry of thisLibraryEntry['generated_images'])
    {
        imageCount += 1;
        const imageName = image_entry.split("/")[2];
        const imageInUpscaleQueue = !!upscaleQueueImagesList.find((image) => image === image_entry);
        const divImageAndButtons = libraryEntry_authorImageAndControls(imageName, image_entry, thisLibraryEntry, imageInUpscaleQueue);
        div.appendChild(divImageAndButtons);
    }
}

// Author the header section for this library Entry
const libraryEntry_authorHeader = (divLibraryEntry, libraryEntry) =>
{
    const hr = document.createElement("hr");
    divLibraryEntry.appendChild(hr);

    const h3 = document.createElement("h3");
    if(divLibraryEntry['negative_prompt'] !== '')
    {
        h3.innerHTML = `<i>${libraryEntry['text_prompt']} [${libraryEntry['negative_prompt']}]</i>`;
    }
    else
    {
        h3.innerHTML = `<i>${libraryEntry['text_prompt']}</i>;`;
    }

    h3.classList.add('h3TextPrompt')
    h3.style.float = 'inline-start';
    divLibraryEntry.appendChild(h3);

    const p = document.createElement("p");
    p.classList.add('parameters-display');
    p.innerHTML = authorParametersListForWeb(libraryEntry);
    divLibraryEntry.appendChild(p);
}

// Author the library entry for this library
const createNewDivLibraryEntry = (libraryEntry, upscaleQueueImagesList) =>
{
    let imageCount = 0;

    const divLibraryEntry = document.createElement('div');
    divLibraryEntry.id = "div_" + libraryEntry['queue_id'];
    divLibraryEntry.dataset['creation_unixtime'] = libraryEntry['creation_unixtime']; // Allowing for insertion in the correct display order
    divLibraryEntry.dataset['text_prompt'] = libraryEntry['text_prompt']; // Allowing for filtering by text prompt
    divLibraryEntry.style.float = 'left';

    // These three functions create and append their various sections to divLibraryEntry
    libraryEntry_authorHeader(divLibraryEntry, libraryEntry);
    if(libraryEntry['generated_images'][0].includes('video.mp4'))
    {
        libraryEntry_authorVideo(divLibraryEntry, libraryEntry);
    }
    else
    {
        libraryEntry_authorMasterImagePlaceHolder(divLibraryEntry, libraryEntry);
        libraryEntry_authorImageDivs(divLibraryEntry, libraryEntry, imageCount, upscaleQueueImagesList);
    }

    return divLibraryEntry;
}

const libraryEntry_authorVideo = (divLibraryEntry, libraryEntry) =>
{
    const video = document.createElement('video');
    video.src = libraryEntry['generated_images'][0];
    video.controls = true;
    video.loop = true;
    video.style.width = 'auto';
    video.style.height = '512px';
    divLibraryEntry.appendChild(video);
}

// Check if there exists an image path in the imageList for the given (non-upscaled) image that includes '_upscaled.'
// in the name. If so, return true. Otherwise, return false.
const checkIfImageAlreadyUpscaled = (imagePath, imageList) =>
{
    let fileFormat = 'png'
    if (imagePath.endsWith('.jpg'))
    {
        fileFormat = 'jpg';
    }
    return !!imageList.find(image => image.includes(`${imagePath.split("/").slice(3).join("/").replace('.' + fileFormat, '')}_upscaled.${fileFormat}`));
}

// scrape the seed value that is embedded in the image's name
const getSeedValueFromImageFileName = (imageFileName) =>
{
    if (imageFileName.includes("blank.") ||
        imageFileName.includes("blank."))
    {
        return '';
    }
    const srcElements = imageFileName.split("/");
    const imageNameSections = srcElements[2].split("-");
    return imageNameSections[3].replace('R', '');
}

// Write a description from th various embedded elements in the image name
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

    const srcElements = imageFileName.split("/");
    const imageNameSections = srcElements[5].split("-");
    const imageNumber = imageNameSections[0]
    const ddimSteps = imageNameSections[1].replace('D', '')
    const scale = imageNameSections[2].replace('S', '');
    const seedValue = imageNameSections[3].replace('R', '');
    const upscaledImage = imageFileName.includes('_upscaled.');
    return `Image #${imageNumber}, DDIM steps: ${ddimSteps}, Scale: ${scale}, Seed: ${seedValue} ${upscaledImage ? ' - (upscaled)' : ''}`;
}

// Create a querystring from the parameters in the library entry that will be used to call the Advanced page.
const authorParametersListForWeb = (libraryItem) =>
{
    let creationDate = new Date(`${libraryItem['creation_unixtime']}`.split(".")[0] * 1000);
    let text = `Images created ${creationDate.toLocaleString()},<br>`;
    text += `Processing took ${libraryItem['time_taken'].toFixed(2)} seconds (${(libraryItem['time_taken'] / libraryItem['generated_images'].length).toFixed((2))} secs/image)`;
    text += `<br>Library folder: ${libraryItem['queue_id']}`;
    text += `<br>parameters:`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;Seed: ${libraryItem['seed']}`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;height: ${libraryItem['height']}px`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;width: ${libraryItem['width']}px`
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;Min DDIM Steps: ${libraryItem['min_ddim_steps']}`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;Max DDIM Steps: ${libraryItem['max_ddim_steps']}`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;scale: ${libraryItem['scale']}`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;downsampling factor: ${libraryItem['downsampling_factor']}`;
    if (libraryItem['original_image_path'] !== '')
    {
        text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;image source: ${libraryItem['original_image_path']},`;
        text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;image strength: ${libraryItem['strength']}`;
    }
    if (libraryItem['error'] !== '')
    {
        text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;<b>Error: ${libraryItem['error']}</b>`;
    }
    return text;

}

// Create link to the Advanced page with the parameters from the library entry
const createLinkToAdvancedPage = (image_src, libraryItem) =>
{
    let textPrompt = libraryItem['text_prompt'];
    if(libraryItem['negative_prompt']!== '')
    {
        textPrompt += ` [${libraryItem['negative_prompt']}]`;
    }
    const urlencoded_image_src = encodeURIComponent(image_src);
    const urlEncodedPrompt = encodeURIComponent(textPrompt);

    let seedValue = getSeedValueFromImageFileName(image_src);
    if (seedValue === '')
    {
        seedValue = libraryItem['seed'];
    }
    const link = `advanced.html?original_image_path=${urlencoded_image_src}&prompt=${urlEncodedPrompt}&seed=${seedValue}&height=${libraryItem['height']}&width=${libraryItem['width']}&min_ddim_steps=${libraryItem['min_ddim_steps']}&max_ddim_steps=${libraryItem['max_ddim_steps']}&ddim_eta=${libraryItem['ddim_eta']}&scale=${libraryItem['scale']}&downsampling_factor=${libraryItem['downsampling_factor']}`;
    return link;
}

// delete an image
const deleteImage = async (img) =>
{
    if (window.confirm(`Are you sure you want to delete image "${img.alt}"?`))
    {
        const rawResponse = await fetch('/image', {
            method: 'DELETE',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                path: img.src.substring(img.src.lastIndexOf('/library') + 1)
            })
        });

        if (rawResponse.status === 200)
        {
            const jsonResult = await rawResponse.json();
            if (jsonResult.success)
            {
                document.getElementById('status').innerText = "Image deleted";
                document.getElementById(`${img.id}`).remove();
            }
            else
            {
                document.getElementById('status').innerText = "Image not deleted";
            }
        }
        else
        {
            document.getElementById('status').innerText = `Sorry, an HTTP error ${rawResponse.status} occurred - check again shortly!`;
        }
    }
}

const processUpscaleQueue = async () =>
{
    if (global_upscaleImageList.length > 0)
    {
        const temp_global_upscaleImageList = JSON.parse(JSON.stringify(global_upscaleImageList));

        // Note: Don't wait for upscale to complete before updating global_upscaleImageList in case
        // this takes longer than the interval (5000ms) between calls to this function.
        upscale(temp_global_upscaleImageList).then();

        // Remove processed images from the list (thus keeping any new requests to be queued)
        global_upscaleImageList.forEach((image, index) =>
        {
            if(temp_global_upscaleImageList.includes(image))
            {
                global_upscaleImageList[index] = ""; // set to empty string to remove
            }
        });
        // Remove empty strings from the list
        global_upscaleImageList = global_upscaleImageList.filter(image => image.length > 0);
    }
}

setInterval(async () => {await processUpscaleQueue()}, 5000);

// Refresh the library index every REFRESH_INTERVAL_SECS seconds
setInterval(function ()  { formatLibraryEntries().then(); }, REFRESH_INTERVAL_SECS * 1000);


