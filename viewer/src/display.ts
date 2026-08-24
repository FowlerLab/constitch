import {convertImage, transformIn, transformOut, intersection} from "./image";
import {onKeyDownCommands} from "./commands";
import {exampleImage} from "./config";
//import {globalColorMat, globalBounds} from "./colors";

export let activeImages = {};
export let globalScreenBox = [0, 0, Math.round(window.innerHeight * 2), Math.round(window.innerWidth * 2)];

let lastUpdate = Date.now();

export function onSettingsChange() {
    lastUpdate = Date.now();
}

export function assignIds(image, begin) {
    image.id = 'image' + begin;
    begin += 1;
    if ('images' in image) {
        for (let subimage of image.images) {
            begin += assignIds(subimage, begin);
        }
    }
    return begin;
}


function setImagePos(image, box) {
    //const box = transformIn(screenBox, {width: window.innerWidth, height: window.innerHeight}, image.box)
    image.canvas.style.transform = 'translate(' + box[1] + 'px, ' + box[0] + 'px)';
    //image.canvas.style.left = box[0] + 'px';
    //image.canvas.style.top = box[1] + 'px';
    //image.canvas.style.width = box[2] + 'px';
    //image.canvas.style.height = box[3] + 'px';
}

function removeImage(image) {
    if (image.id in activeImages) {
        delete activeImages[image.id];
        if (image.rendered) {
            document.getElementById('fetched-images').appendChild(image.canvas);
        } else {
            image.canvas.remove();
            delete image.canvas
            image.loaded = false;
        }
    }

    if ('images' in image) {
        for (let subimage of image.images) removeImage(subimage);
    }
}

export function drawMinimap(baseImage, image, canvas, ctx) {
    if (!canvas) {
        canvas = document.querySelector('.minimap canvas');
        ctx = canvas.getContext('2d');
        ctx.strokeStyle = 'lightblue';
        ctx.lineWidth = 2;
    }

    if ('images' in image && image.images.length != 0) {
        for (let subimage of image.images) {
            drawMinimap(baseImage, subimage, canvas, ctx);
        }

        if (!image.trueImage) {
            return;
        }
    }

    if (image.imageTile) {
        return;
    }

    const box = transformIn(baseImage.box, {width: canvas.width - 2, height: canvas.height - 2}, image.box);
    ctx.strokeRect(box[1] + 1, box[0] + 1, box[3], box[2]);
}

export function updateMinimap(screenBox, baseImage) {
    console.log('updating');
    const viewbox = document.querySelector('.minimap .viewbox');
    const viewboxDashed = document.querySelector('.minimap .viewbox-dashed');

    const box = transformIn(baseImage.box, {width: 100, height: 100}, screenBox);
    const solidBox = intersection([-5, -5, 110, 110], box);
    solidBox[2] = Math.max(0, solidBox[2]);
    solidBox[3] = Math.max(0, solidBox[3]);
    viewbox.style.top = solidBox[0] + '%';
    viewbox.style.left = solidBox[1] + '%';
    viewbox.style.height = solidBox[2] + '%';
    viewbox.style.width = solidBox[3] + '%';

    const clipBox = intersection([0, 0, 100, 100], box);
    clipBox[2] = Math.max(0, clipBox[2]);
    clipBox[3] = Math.max(0, clipBox[3]);
    //const clipBox = [Math.max(0, box[0]), Math.max(0, box[1]), box[2] + Math.min(0, box[0]), box[3] + Math.min(0, box[1])];
    //clipBox[2] = Math.min(100 - box[0], box[2]);
    //clipBox[3] = Math.min(100 - box[1], box[3]);
    viewboxDashed.style.top = clipBox[0] + '%';
    viewboxDashed.style.left = clipBox[1] + '%';
    viewboxDashed.style.height = clipBox[2] + '%';
    viewboxDashed.style.width = clipBox[3] + '%';
    console.log(clipBox);
    console.log(viewbox.style.left);
}

export function updateActiveImages(screenBox, colormat, bounds, image, parentImage) {
    const radius = 0;//-100;
    const bigScreenBox = [screenBox[0] - radius, screenBox[1] - radius, screenBox[2] + radius * 2, screenBox[3] + radius * 2];

    //for (let i = 0; i < images.length; i ++) {

        //const image = images[i];

    const curSection = intersection(image.box, bigScreenBox);

    //console.log(i, image.box, curSection, bigScreenBox)
    image.visible = curSection[2] > 0 && curSection[3] > 0
    image.pixelSize = image.box[2] / image.dims[0] * window.innerHeight / screenBox[2];
    //console.log('pixelsize', image.pixelSize)

    if (!image.visible) {
        console.log('  not visible', image.id);
        console.log('     ', curSection, bigScreenBox, image.box);
        if (image.loaded && image.id in activeImages) {
            removeImage(image);
            //delete activeImages[image.id];
            //document.getElementById('fetched-images').appendChild(image.canvas);

            //if ('images' in image && image.images.length != 0) {
                //updateActiveImages(screenBox, colormat, bounds, image.images, image)
            //}
        }
        return;
    }

    if (!('url' in image)) {
        if ('images' in image) {
            for (let subimage of image.images) {
                updateActiveImages(screenBox, colormat, bounds, subimage, image);
            }
        }
        return;
    }

    console.log('  pixelSize', image.pixelSize);
    if (image.pixelSize > 1 && 'images' in image && image.images.length != 0) {
        let meanPixelSize = 0;
        let isActiveChild = false;
        for (let subImage of image.images) {
            meanPixelSize += subImage.box[2] / subImage.dims[0] * window.innerHeight / screenBox[2];
            isActiveChild = isActiveChild || (subImage.id in activeImages);
        }
        meanPixelSize /= image.images.length;

        //if (meanPixelSize > 1 || 1 / meanPixelSize - 1 < 1 - 1 / image.pixelSize) {
        if (meanPixelSize > 1 || isActiveChild) {
            console.log('  pixel size ', image.pixelSize, meanPixelSize, 'too big', image.id);
            if (image.loaded && image.id in activeImages) {
                delete activeImages[image.id];
                document.getElementById('fetched-images').appendChild(image.canvas);
            }
            for (let subimage of image.images) {
                updateActiveImages(screenBox, colormat, bounds, subimage, image);
            }
            return;
        }
    }

    if (!image.loaded) {
        console.log('  loading image', image.id);
        const img = document.createElement('img');

        const canvas = document.createElement('canvas');
        canvas.height = image.dims[0];
        canvas.width = image.dims[1];
        canvas.classList.add('image');
        canvas.id = image.id
        image.canvas = canvas;

        if (!('images' in image) || image.images.length == 0) {
            canvas.style.imageRendering = 'pixelated';
        }

        const ctx = canvas.getContext('2d');

        // prerender with prev canvas contents
        if (parentImage != undefined && 'canvas' in parentImage) {
            const srcBox = transformIn(parentImage.box, {width: parentImage.dims[0], height: parentImage.dims[1]}, image.box);
            //console.log('transform in', ...parentImage.box, parentImage.width, parentImage.height, ...image.box)
            console.log('copying', parentImage.canvas, ...srcBox, 0, 0, canvas.width, canvas.height);
            //ctx.drawImage(parentImage.canvas, ...srcBox, 0, 0, canvas.width, canvas.height);
            ctx.drawImage(parentImage.canvas, srcBox[1], srcBox[0], srcBox[3], srcBox[2], 0, 0, canvas.height, canvas.width);
        } else {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        setImagePos(image, image.box);
        image.canvas.style.height = image.box[2] + 'px';
        image.canvas.style.width = image.box[3] + 'px';
        const begin = Date.now()
        image.rendered = false;

        img.onload = () => {
            const mid = Date.now();
            convertImage(img, image, colormat, bounds, canvas);
            console.log('    finished loading', mid - begin, Date.now() - mid);
            image.lastUpdate = Date.now();
            image.rendered = true;
        };

        const loadingSetting = document.querySelector('input[name="loading-behaviour"]:checked').value;
        if (loadingSetting == 'auto' || image.loading == 'auto') {
            img.src = image.url;
        } else {
            const url = image.url;
            const callback = (event) => {
                if (event.shiftKey) {
                    img.src = url;
                    canvas.removeEventListener('click', callback);
                }
            };
            canvas.addEventListener('click', callback);
        }

        image.loaded = true;
        image.lastUpdate = Date.now();
    } else if (image.lastUpdate < lastUpdate) {
        console.log('  rerendering image', image.id);
        setTimeout(() => {
            convertImage(null, image, colormat, bounds, image.canvas);
            image.lastUpdate = Date.now();
        }, 1);
        image.lastUpdate = Date.now();
    }

    if (!(image.id in activeImages)) {
        console.log('  adding back', image.id);
        activeImages[image.id] = image;
        //const div = document.createElement('div')
        //div.appendChild(image.canvas)
        //document.getElementById('axes').appendChild(div);
        document.getElementById('axes').appendChild(image.canvas);

        if ('images' in image) {
            for (let subimage of image.images) removeImage(subimage);
        }
    }

    // If this image was rendered don't go deeper
    //if ('images' in image && image.images.length != 0) {
        //updateActiveImages(screenBox, colormat, bounds, image.images, image)
    //}
    //}
}

function updatePoses(screenBox) {
    for (let image of Object.values(activeImages)) {
        setImagePos(image, transformIn(screenBox, {width: window.innerHeight, height: window.innerWidth}, image.box));
    }
}

function updateTransform(screenBox) {
    document.getElementById('axes').style.transform = (
        'scale(' + (window.innerHeight / screenBox[2])
        + ') translate(' + (-screenBox[1])
        + 'px, ' + (-screenBox[0]) + 'px)');
}


export function resetScreenBox() {
    const aspectRatio = window.innerHeight / window.innerWidth;
    globalScreenBox = [...exampleImage.box];
    globalScreenBox[2] = Math.max(exampleImage.box[2], exampleImage.box[3] * aspectRatio);
    globalScreenBox[3] = Math.max(exampleImage.box[3], exampleImage.box[2] / aspectRatio);
    updateTransform(globalScreenBox);
}

let timeoutId = 0;

export function onScreenChange() {
    if (timeoutId != 0) {
        window.clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
        console.log('Updating active images', globalScreenBox);
        updateMinimap(globalScreenBox, exampleImage);
        updateActiveImages(globalScreenBox, exampleImage.colorMat, exampleImage.bounds, exampleImage);
    }, 500);
}

export function onViewChange() {
    if (timeoutId != 0) {
        window.clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
        console.log('Updating active images', globalScreenBox);
        updateMinimap(globalScreenBox, exampleImage);
        //const loadingSetting = document.querySelector('input[name="loading-behaviour"]:checked').value;
        //if (loadingSetting == 'auto')
        updateActiveImages(globalScreenBox, exampleImage.colorMat, exampleImage.bounds, exampleImage);
    }, 500);
}

export var undoList = [];

export function undoMove() {
    if (undoList.length == 0) return;

    const prevstate = undoList.pop();
    console.log(prevstate)
    for (let canvasid of Object.keys(prevstate)) {
        activeImages[canvasid].box[0] = prevstate[canvasid][0]
        activeImages[canvasid].box[1] = prevstate[canvasid][1]
        setImagePos(activeImages[canvasid], activeImages[canvasid].box)
    }
    onViewChange();
    updateTransform(globalScreenBox);
}

export function undoAllMoves() {
    while (undoList.length > 0) {
        undoMove();
    }
}


//document.getElementById('plot').addEventListener('click', (event) => {
    //const loadingSetting = document.querySelector('input[name="loading-behaviour"]:checked').value;
    //onScreenChange();
//});

document.getElementById('plot').addEventListener('mousedown', (event) => {
    const tool = document.querySelector('input[name="tool"]:checked').value

    if (tool == 'move' || tool == 'select') {
        if (event.target.classList.contains('selected') && (tool == 'select' || event.ctrlKey)) {
            event.target.classList.remove('selected')
            return;
        }
        if (!event.ctrlKey && tool != 'select') {
            document.querySelectorAll('.image.selected').forEach(elem => elem.classList.remove('selected'))
        }
        if (event.target.tagName == 'CANVAS') {
            event.target.classList.add('selected')
        }
    }
})


document.getElementById('plot').addEventListener('mousemove', (event) => {
    if (event.buttons != 1) return;
    const tool = document.querySelector('input[name="tool"]:checked').value

    if (tool == 'pan') {
        globalScreenBox[0] += -event.movementY * (globalScreenBox[2] / window.innerHeight);
        globalScreenBox[1] += -event.movementX * (globalScreenBox[3] / window.innerWidth);
    } else if (tool == 'move') {
        const prevstate = {};
        for (const canvas of document.querySelectorAll('.image.selected')) {
            const image = activeImages[canvas.id];
            prevstate[canvas.id] = [image.box[0], image.box[1]];
            image.box[0] += event.movementY * (globalScreenBox[2] / window.innerHeight);
            image.box[1] += event.movementX * (globalScreenBox[3] / window.innerWidth);
            setImagePos(image, image.box)
        }
        undoList.push(prevstate);
    }
    onViewChange();
    updateTransform(globalScreenBox);
})

document.getElementById('plot').addEventListener('wheel', (event) => {
    //let deltaScale = 1;
    //if (event.deltaY < 0) {
        //deltaScale = 7/8;
    //}
    //if (event.deltaY > 0) {
        //deltaScale = 9/8;
    //}
    const deltaScale = Math.exp(event.deltaY * 0.0015);
    if (deltaScale != 1) {
        globalScreenBox[0] += event.clientY * (1 - deltaScale) * (globalScreenBox[2] / window.innerHeight);
        globalScreenBox[1] += event.clientX * (1 - deltaScale) * (globalScreenBox[3] / window.innerWidth);
        globalScreenBox[2] *= deltaScale;
        globalScreenBox[3] *= deltaScale;
        //scaleMidpoint = [event.clientX, event.clientY];
        onViewChange();
        updateTransform(globalScreenBox);
    }
})

document.addEventListener('keydown', (event) => {
    if (!onKeyDownCommands(event)) return;
    if (event.key == 'g') {
        resetScreenBox();
        onViewChange();
    }
})

let windowSize = [window.innerHeight, window.innerWidth];

window.addEventListener('resize', (event) => {
    globalScreenBox[2] *= window.innerHeight / windowSize[0];
    globalScreenBox[3] *= window.innerWidth / windowSize[1];
    windowSize = [window.innerHeight, window.innerWidth];
    onViewChange();
    updateTransform(globalScreenBox);
})
