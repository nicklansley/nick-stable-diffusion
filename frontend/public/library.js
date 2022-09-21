let library = [];
let autoRefreshId;
let defaultUpscaleFactor = 4;

const listLibrary = async () =>
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
        return await rawResponse.json();
    }
    else
    {
        document.getElementById('status').innerText = `Sorry, an HTTP error ${rawResponse.status} occurred - check again shortly!`;
        return [];
    }
}


const upscale = async (image_list) =>
{
    const upscaleObj = {
        "image_list": image_list,
        "upscale_factor": defaultUpscaleFactor
    }

    let rawResponse;
    document.getElementById('status').innerText = "Sending Upscale request...";

    try
    {
        rawResponse = await fetch('/upscale', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(upscaleObj)
        });
    }
    catch (e)
    {
        document.getElementById('status').innerText = "Sorry, service offline for upscaling request";
        return false;
    }

    if (rawResponse.status === 200)
    {
        document.getElementById('status').innerText = "Upscale request sent to queue";
        return await rawResponse.json();
    }
    else
    {
        document.getElementById('status').innerText = `Sorry, an HTTP error ${rawResponse.status} occurred when upscaling - check again shortly!`;
        return [];
    }
}


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

const formatLibraryEntries = async () =>
{
    let imageCount = 0;
    let libraryEntryCount = 0;

    const upscaleQueueImagesList = await getQueueAndListUpscaleRequests();

    // retrieve the library and sort by descending creation_unixtime
    library = await listLibrary();
    library = library.sort((a, b) => a.creation_unixtime > b.creation_unixtime ? -1 : 1);

    //get the searchText value (if any)
    const searchText = document.getElementById('searchText').value;

    // Clear the listing
    const output = document.getElementById("output");
    output.innerHTML = "";

    for(const libraryItem of library)
    {
        const divLibraryItem = document.createElement('div');
        divLibraryItem.style.float = 'left';

        if (searchText.length === 0 || libraryItem['text_prompt'].toLowerCase().includes(searchText.toLowerCase()))
        {
            libraryEntryCount += 1;
            const hr = document.createElement("hr");
            divLibraryItem.appendChild(hr);

            const h3 = document.createElement("h3");
            h3.innerHTML = `<i>${libraryItem['text_prompt']}</i>`;
            h3.style.float = 'inline-start';
            divLibraryItem.appendChild(h3);

            const p = document.createElement("p");
            p.classList.add('parameters-display');
            p.innerHTML = authorParametersListForWeb(libraryItem);
            divLibraryItem.appendChild(p);


            // Add master image placeholder
            const masterImage = document.createElement("img");
            if (libraryItem['generated_images'].length > 0)
            {
                if (libraryItem['generated_images'][0].includes('00-original.'))
                {
                    masterImage.src = libraryItem['generated_images'][1]; // the second image is the first generated image when using an input image
                }
                else
                {
                    masterImage.src = libraryItem['generated_images'][0]; // the first image is the first generated image
                }

                // Master image for group
                masterImage.id = `master_image_${libraryItem['queue_id']}`;
                masterImage.alt = libraryItem['text_prompt'];
                masterImage.height = libraryItem['height'];
                masterImage.width = libraryItem['width'];
                masterImage.style.zIndex = "0";
                divLibraryItem.appendChild(masterImage);

                // Caption for master image
                const masterImageCaption = document.createElement("p")
                masterImageCaption.id = `master_image_caption_${libraryItem['queue_id']}`;
                masterImageCaption.style.float = 'inline-start';
                divLibraryItem.appendChild(masterImageCaption);


            }

            for(const image_entry of libraryItem['generated_images'])
            {
                imageCount += 1;
                const imageName = image_entry.split("/")[2];
                const imageInUpscaleQueue = !!upscaleQueueImagesList.find((image) => image === image_entry);

                const divImageAndButtons = document.createElement("div");
                divImageAndButtons.classList.add('divImage');
                if (imageName.includes('_upscaled.'))
                {
                    divImageAndButtons.style.borderColor = "gold";
                    divImageAndButtons.style.borderWidth = "5px";
                }

                const image = document.createElement("img");
                image.id = imageName.split('.')[0];
                image.src = image_entry;
                image.alt = libraryItem['text_prompt'];
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
                            const imageList = [];
                            const imageRelativePath = image.src.split("/").slice(3).join("/");
                            imageList.push(imageRelativePath);
                            upscale(imageList);
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
                divLibraryItem.appendChild(divImageAndButtons);

            }
            output.appendChild(divLibraryItem)

        }
    }
    const dateNow = new Date();
    document.getElementById('status').innerText = `Updated ${dateNow.toLocaleString()} - Found ${imageCount} images within ${libraryEntryCount} library entries`;
}


const checkIfImageAlreadyUpscaled = (imagePath, imageList) =>
{
    let fileFormat = 'png'
    if(imagePath.endsWith('.jpg'))
    {
        fileFormat = 'jpg';
    }
    return !!imageList.find(image => image.includes(`${imagePath.split("/").slice(3).join("/").replace('.' + fileFormat, '')}_upscaled.${fileFormat}`));
}

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

const authorDescriptionFromImageFileName = (imageFileName) =>
{
    if (imageFileName.includes("blank."))
    {
        return '';
    }
    if( imageFileName.includes("original."))
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

const createLinkToAdvancedPage = (image_src, libraryItem) =>
{
    const urlencoded_image_src = encodeURIComponent(image_src);
    const urlEncodedPrompt = encodeURIComponent(libraryItem['text_prompt']);
    let seedValue = getSeedValueFromImageFileName(image_src);
    if (seedValue === '')
    {
        seedValue = libraryItem['seed'];
    }
    const link = `advanced.html?original_image_path=${urlencoded_image_src}&prompt=${urlEncodedPrompt}&seed=${seedValue}&height=${libraryItem['height']}&width=${libraryItem['width']}&min_ddim_steps=${libraryItem['min_ddim_steps']}&max_ddim_steps=${libraryItem['max_ddim_steps']}&ddim_eta=${libraryItem['ddim_eta']}&scale=${libraryItem['scale']}&downsampling_factor=${libraryItem['downsampling_factor']}`;
    return link;
}


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

const setAutoRefresh = async () =>
{
    const checkBox = document.getElementById('autoRefresh');
    if (checkBox.checked)
    {
        await listLibrary();
        autoRefreshId = setInterval(function ()
        {
            formatLibraryEntries().then();
        }, 10000);
    }
    else
    {
        clearInterval(autoRefreshId);
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
