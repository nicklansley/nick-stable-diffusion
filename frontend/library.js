let library = [];
let autoRefreshId;

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

const retrieveImages = async () =>
{
    let imageCount = 0;
    let libraryEntryCount = 0;

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
        if (searchText.length === 0 || libraryItem['text_prompt'].includes(searchText))
        {
            libraryEntryCount += 1;
            const hr = document.createElement("hr");
            document.getElementById("output").appendChild(hr);

            const h3 = document.createElement("h3");
            let creationDate = new Date(`${libraryItem['creation_unixtime']}`.split(".")[0] * 1000);
            h3.innerHTML = `<i>${libraryItem['text_prompt']}</i>`;
            document.getElementById("output").appendChild(h3);

            const p = document.createElement("p");
            p.classList.add('parameters-display');
            p.innerHTML = authorParametersListForWeb(libraryItem);
            document.getElementById("output").appendChild(p);

            const masterImage = document.createElement("img");
            masterImage.src =  libraryItem['generated_images'][0]; // the first image is the current master image
            masterImage.id = `master_image_${libraryItem['queue_id']}`;
            masterImage.alt = libraryItem['text_prompt'];
            masterImage.height = libraryItem['height'];
            masterImage.width = libraryItem['width'];
            masterImage.style.zIndex = "0";
            document.getElementById("output").appendChild(masterImage);
            document.getElementById("output").appendChild(document.createElement("br"));

            for(const image_entry of libraryItem['generated_images'])
            {
                imageCount += 1;
                const imageName = image_entry.split("/")[2];

                const image = document.createElement("img");
                image.src = image_entry;
                image.id = imageName.split('.')[0];
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
                // image.style.zIndex = "0";
                image.onclick = function ()
                {
                    window.open(`${createLinkToAdvancedPage(image_entry, libraryItem)}`, '_blank');
                }
                image.onmouseover = function ()
                {
                    this.style.transform = `scale(1.5)`;
                    this.style.transform = `translate(50px,0px)`;
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

                image.oncontextmenu = function (ev)
                {
                    ev.preventDefault();
                    deleteImage(this);
                };

                output.appendChild(image);

                if (imageCount % 6 === 0)
                {
                    output.appendChild(document.createElement("br"));
                }
            }

        }
    }
    const dateNow = new Date();
    document.getElementById('status').innerText = `Updated ${dateNow.toLocaleString()} - Found ${imageCount} images within ${libraryEntryCount} library entries`;
}

const authorParametersListForWeb = (libraryItem) =>
{
    let creationDate = new Date(`${libraryItem['creation_unixtime']}`.split(".")[0] * 1000);
    let text = `Images created ${creationDate.toLocaleString()},<br>`;
    text += `Processing took ${libraryItem['time_taken'].toFixed(2)} seconds (${(libraryItem['time_taken'] / libraryItem['generated_images'].length).toFixed((2))} secs/image)`;
    text += `<br>parameters:`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;Seed: ${libraryItem['seed']}`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;height: ${libraryItem['height']}px`;
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;width: ${libraryItem['width']}px`
    text += `<br>&nbsp;&nbsp;&nbsp;&nbsp;DDIM Steps: ${libraryItem['ddim_steps']}`;
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
    const link = `advanced.html?original_image_path=${urlencoded_image_src}&prompt=${urlEncodedPrompt}&seed=${libraryItem['seed']}&height=${libraryItem['height']}&width=${libraryItem['width']}&ddim_steps=${libraryItem['ddim_steps']}&ddim_eta=${libraryItem['ddim_eta']}&scale=${libraryItem['scale']}&downsampling_factor=${libraryItem['downsampling_factor']}`;
    return link;
}


const deleteImage = async (img) =>
{
    if (window.confirm(`Are you sure you want to delete image "${img.alt}"?`))
    {
        const rawResponse = await fetch('/deleteimage', {
            method: 'POST',
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
            retrieveImages().then();
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
