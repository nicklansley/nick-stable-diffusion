let global_imageCanvas = null;
let global_imageCtx = null;
let global_maskCanvas = null;
let global_maskCtx = null;
let global_imageBackupsList = [];
let global_maskBackupsList = [];
let global_lineWidth = 5;
let global_inStrokeMode = true;

const addEventListenersToImageCanvas = () =>
{
    global_imageCanvas.addEventListener('dragover', (e) =>
    {
        e.preventDefault();
        e.stopPropagation();
        global_imageCanvas.classList.add('drag-over');
    });

    global_imageCanvas.addEventListener('dragleave', (e) =>
    {
        e.preventDefault();
        e.stopPropagation();
        global_imageCanvas.classList.remove('drag-over');
    });

    global_imageCanvas.addEventListener('drop', (e) =>
    {
        e.preventDefault();
        e.stopPropagation();
        global_imageCanvas.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file)
        {
            const reader = new FileReader();
            reader.onload = (e) =>
            {
                const img = new Image();
                img.onload = () =>
                {
                    // Make the image fit the canvas
                    const ratio = Math.min(global_imageCanvas.width / img.width, global_imageCanvas.height / img.height);
                    const width = img.width * ratio;
                    const height = img.height * ratio;
                    global_imageCtx.drawImage(img, 0, 0, width, height);

                    // Save the canvas state as our initial 'backup' state
                    global_imageBackupsList = [global_imageCanvas.toDataURL()]

                    // Now load the 512px black square into the mask canvas to initialise it
                    const maskImg = new Image();
                    maskImg.onload = () =>
                    {
                        // Initialise mask with a black square
                        global_maskCtx.drawImage(maskImg, 0, 0, width, height);
                        global_maskBackupsList = [global_maskCanvas.toDataURL()]
                    }
                    maskImg.src = "512px_black_square.png";

                }

                img.src = e.target.result;
            }
            reader.readAsDataURL(file);
        }
    });

    // add an event listener to the canvas to allow drawing with the mouse
    global_imageCanvas.addEventListener('mousedown', (e) =>
    {
        global_imageCtx.beginPath();
        global_imageCtx.moveTo(e.offsetX, e.offsetY);

        global_maskCtx.beginPath();
        global_maskCtx.moveTo(e.offsetX, e.offsetY);
    });

    global_imageCanvas.addEventListener('mousemove', (e) =>
    {
        if (e.buttons === 1)
        {
            // Set the line width and color
            global_imageCtx.lineWidth = global_lineWidth;
            global_imageCtx.lineCap = 'round';
            global_imageCtx.strokeStyle = 'black';

            global_maskCtx.lineWidth = global_lineWidth;
            global_maskCtx.lineCap = 'round';
            global_maskCtx.strokeStyle = 'white';

            global_imageCtx.lineTo(e.offsetX, e.offsetY);
            global_maskCtx.lineTo(e.offsetX, e.offsetY);

            if (global_inStrokeMode)
            {
                global_imageCtx.stroke();
                global_maskCtx.stroke();
            }
            else
            {
                global_imageCtx.fill();
                global_maskCtx.fill();
            }
        }
    });

    global_imageCanvas.addEventListener('mouseup', (e) =>
    {
        global_imageCtx.closePath();
        global_maskCtx.closePath();

        // Push the current canvas state to the list of backups
        global_imageBackupsList.push(global_imageCanvas.toDataURL());
        global_maskBackupsList.push(global_maskCanvas.toDataURL());
    });
}

const inPaintSetup = () =>
{
    global_imageCanvas = document.getElementById("imagecanvas");
    global_imageCtx = global_imageCanvas.getContext("2d");

    global_maskCanvas = document.getElementById("maskcanvas");
    global_maskCtx = global_maskCanvas.getContext("2d");

    addEventListenersToImageCanvas();
}

const inPaintChangeLineWidth = (e) =>
{
    const lineWidthControl = document.getElementById("line_width");
    global_lineWidth = lineWidthControl.value;
    document.getElementById("display_line_width").innerHTML = global_lineWidth + "px";
}


const inPaintModeStrokeFill = (button) =>
{
    global_inStrokeMode = !global_inStrokeMode;
    button.innerHTML = global_inStrokeMode ? "Stroke mode - switch to Fill mode" : "Fill mode - switch to Stroke mode";
}


const _undoImage = () =>
{
    //Restore the last saved canvas state
    const img = new Image();
    img.onload = () =>
    {
        global_imageCtx.drawImage(img, 0, 0);
    }

    // Remove the last saved canvas state
    if (global_imageBackupsList.length > 1)
    {
        img.src = global_imageBackupsList[global_imageBackupsList.length - 2];
        // Remove the last saved canvas state
        global_imageBackupsList.pop();
    }

}

const _undoMask = () =>
{
    //Restore the last saved canvas state
    const img = new Image();
    img.onload = () =>
    {
        global_maskCtx.drawImage(img, 0, 0);
    }

    // Remove the last saved canvas state
    if (global_maskBackupsList.length > 1)
    {
        img.src = global_maskBackupsList[global_maskBackupsList.length - 2];
        // Remove the last saved canvas state
        global_maskBackupsList.pop();
    }

}


const inpaintGo = async () =>
{
        document.getElementById('status').innerText = `Creating inpainted images...`
        document.getElementById('buttonGo').innerText = `Creating inpainted images...`;
        document.getElementById('buttonGo').enabled = false;

        global_imagesRequested = parseInt(document.getElementById('num_images').value);

        //If the timer is already running and the button is clicked again, the user is sending a new request
        //to be added to the queue. So we need to stop the countdown timer, as the countdown is no longer
        //applicable to this new current request.
        if (global_countdownTimerIntervalId)
        {
            clearInterval(global_countdownTimerIntervalId);
            global_countdownTimerIntervalId = null;
        }

        const data = prepareRequestData();
        // Remove a couple of properties that we don't need to send to the server
        delete data['prompt']
        delete data['negative_prompt']
        data['format'] = 'inpaint';
        const rawResponse = await sendPromptRequest(data);
        await processPromptRequestResponse(rawResponse);
}

// add a function called 'undo' that revreses the last mouse movement drawing
const inPaintUndo = () =>
{
    _undoImage();
    _undoMask();
}

