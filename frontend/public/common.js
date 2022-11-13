// Set the dark mode on/off based on the value in local storage
const setDarkModeFromLocalStorage = () =>
{
    if (localStorage.getItem("dark-mode") === "Y")
    {
        document.body.classList.add('dark-mode');
    }
}

// Switch on/off dark mode for this page
const toggleDarkMode = () =>
{
    let element = document.body;
    element.classList.toggle('dark-mode');
    localStorage.setItem("dark-mode", element.classList.contains('dark-mode') ? "Y" : "N");
}


// Upscale an image
const upscale = async (upscaleImageList) =>
{
    if(global_upscaleImageList.length === 0)
    {
        return;
    }
    const upscaleObj = {
        "image_list": upscaleImageList.filter(image => image.length > 0),
        "upscale_factor": DEFAULT_UPSCALE_FACTOR
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


