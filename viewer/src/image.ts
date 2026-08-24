

export function convertImage(imagetag, info, colormat, bounds, canvastag) {
    const canvasid = 'rawcanvas' + info.id;
    let canvas = document.getElementById(canvasid);
    let ctx;

    if (!canvas) {
        canvas = document.createElement('canvas');

        canvas.id = canvasid;
        canvas.width = imagetag.naturalWidth || imagetag.width;
        canvas.height = imagetag.naturalHeight || imagetag.height;

        document.getElementById('raw-canvases').appendChild(canvas);

        ctx = canvas.getContext('2d');
        ctx.drawImage(imagetag, 0, 0, canvas.width, canvas.height);

    } else {
        ctx = canvas.getContext('2d');
    }

    canvastag.width = info.dims[1];
    canvastag.height = info.dims[0];

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const destCtx = canvastag.getContext('2d');
    const destData = destCtx.getImageData(0, 0, canvastag.width, canvastag.height)

    const numChannels = (info.shape.length == 2) ? 1 : info.shape[0];
    const channelOffset = destData.data.length;

    for (let i = 0; i < destData.data.length; i += 4) {
        const finalColor = [0, 0, 0];
        for (let j = 0; j < numChannels; j ++) {
            let value = imageData.data[i+channelOffset*j];
            value |= imageData.data[i+channelOffset*j+1] << 8;
            value = (value - bounds[j][0]) / (bounds[j][1] - bounds[j][0]) * 255;
            finalColor[0] += value * colormat[0][j];
            finalColor[1] += value * colormat[1][j];
            finalColor[2] += value * colormat[2][j];
        }
        destData.data[i] = finalColor[0];
        destData.data[i+1] = finalColor[1];
        destData.data[i+2] = finalColor[2];
        destData.data[i+3] = 255;
    }
    destCtx.putImageData(destData, 0, 0);
}


export function transformIn(box, image, innerBox) {
    return [
        (innerBox[0] - box[0]) * image.width / box[2],
        (innerBox[1] - box[1]) * image.height / box[3],
        innerBox[2] * image.width / box[2],
        innerBox[3] * image.height / box[3],
    ];
}


export function transformOut(box, image, pixelBox) {
    return [
        box[0] + pixelBox[0] * box[2] / image.width,
        box[1] + pixelBox[1] * box[3] / image.height,
        pixelBox[2] * box[2] / image.width,
        pixelBox[3] * box[3] / image.height,
    ];
}


export function overlapping(box1, box2) {
    return ((box1[0] + box1[2] > box2[0] || box2[0] + box2[2] > box1[0])
         && (box1[1] + box1[3] > box2[1] || box2[1] + box2[3] > box1[1]));
}


export function intersection(box1, box2) {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[0] + box1[2], box2[0] + box2[2]);
    const y2 = Math.min(box1[1] + box1[3], box2[1] + box2[3]);
    return [x1, y1, x2 - x1, y2 - y1];
}
