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

    if(rawResponse.status === 200)
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

    for (const libraryItem of library)
    {
        if(searchText.length === 0 || libraryItem['text_prompt'].includes(searchText))
        {
            if(libraryItem['generated_images'].length > 0)
            {
                libraryEntryCount += 1;
                const hr = document.createElement("hr");
                document.getElementById("output").appendChild(hr);

                const h3 = document.createElement("h3");
                let creationDate = new Date(`${libraryItem['creation_unixtime']}`.split(".")[0] * 1000);
                h3.innerHTML = `<i>${libraryItem['text_prompt']}</i><br>`;
                h3.innerHTML += `<small>${creationDate.toLocaleString()}</small><br>`;
                h3.innerHTML += `<small>parameters: Seed: ${libraryItem['seed']}, height: ${libraryItem['height']}px, width ${libraryItem['width']}px, DDIM Steps: ${libraryItem['ddim_steps']}, DDIM ETA: ${libraryItem['ddim_eta']}, scale: ${libraryItem['scale']}, downsampling factor: ${libraryItem['downsampling_factor']}</small><br>`;
                if(libraryItem['error'] !== '')
                {
                    h3.innerHTML += `<small>Error: <b>${libraryItem['error']}</b></small><br>`;
                }
                h3.innerHTML += `<small>processing took ${libraryItem['process_time_secs'].toFixed(2)} seconds (${(libraryItem['process_time_secs'] / libraryItem['generated_images'].length).toFixed((2))} secs/image)</small>`;
                document.getElementById("output").appendChild(h3);

                for (const image_entry of libraryItem['generated_images'])
                {
                    imageCount += 1;
                    const imageName = image_entry.split("/")[2];
                    const image = document.createElement("img");
                    image.src = image_entry;
                    image.id = imageName.split('.')[0];
                    image.alt = libraryItem['text_prompt'];
                    image.height = 150;
                    image.width = 150;


                    // Add data-image-details attribute to image using the
                    // libraryItem object with generated_images list deleted.
                    const dataImageDetails = JSON.parse(JSON.stringify(libraryItem));
                    delete dataImageDetails['generated_images'];
                    dataImageDetails.path = image_entry;
                    image.setAttribute('data-image-details', JSON.stringify(dataImageDetails));
                    image.onclick = function ()
                    {
                        if(this.nodeName === "IMG")
                        {
                            if(this.style.transform.includes("scale(3)"))
                            {
                                this.style.transform = "scale(1)";
                                this.style.transform += "translate(0px,0px)";
                                this.style.transition = "transform 0.25s ease";
                                this.style.zIndex = this.getAttribute('old_z');
                            }
                            else
                            {
                                this.style.transform = "scale(3)";
                                this.style.transform += "translate(50px,0px)";
                                this.style.transition = "transform 0.25s ease";
                                this.setAttribute('old_z', this.style.zIndex);
                                this.style.zIndex = "-1";
                            }
                        }
                    };
                    image.ondblclick = function ()
                    {
                        deleteImage(this);
                    };
                    output.appendChild(image);
                    if(imageCount % 6 === 0)
                    {
                        output.appendChild(document.createElement("br"));
                    }                }
            }
        }
    }
    const dateNow = new Date();
    document.getElementById('status').innerText = `Updated ${dateNow.toLocaleString()} - Found ${imageCount} images within ${libraryEntryCount} library entries`;
}


const deleteImage = async (img) =>
{
    const jsonData = img.getAttribute('data-image-details');
    const data = JSON.parse(jsonData.replaceAll("&quot;", "\""));

    if(window.confirm(`Are you sure you want to delete image "${data.text_prompt}"?`))
    {
        const rawResponse = await fetch('/deleteimage', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                path: data.path
            })
        });

        if(rawResponse.status === 200)
        {
            const jsonResult = await rawResponse.json();
            if(jsonResult.success)
            {
                document.getElementById('status').innerText = "Image deleted";
                document.getElementById(`${img.id}`).remove();
            }
            else
            {
                document.getElementById('status').innerText = "Image not deleted";
            }
        } else
        {
            document.getElementById('status').innerText = `Sorry, an HTTP error ${rawResponse.status} occurred - check again shortly!`;
        }
    }
}

const setAutoRefresh = async () =>
{
    const checkBox = document.getElementById('autoRefresh');
    if(checkBox.checked)
    {
        await listLibrary();
        autoRefreshId = setInterval(function ()
        {
            retrieveImages().then();
        }, 10000);
    } else
    {
        clearInterval(autoRefreshId);
    }

}
