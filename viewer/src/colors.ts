import {setColorMat, setBounds} from './settings';
import {exampleImage} from './config';

/*
export const globalColorMat = [
    [1/50, 0, 1/50, 0, 0, 1/50],
    [1/50, 1/50, 0, 1/50, 1/50, 0],
    [1/50, 0, 1/50, 1/50, 0, 0],
];

export const globalBounds = [
    [0, 5000],
    [0, 5000],
    [0, 5000],
    [0, 5000],
    [0, 5000],
    [0, 5000],
];
*/

export function defaultBounds(image) {
    image.bounds = []
    for (let i = 0; i < exampleImage.numChannels; i ++) {
        image.bounds.push([0, 5000]);
    }
    setBounds(5000);
}

export function defaultColorMat(image) {
    image.colorMat = [
        new Array(image.numChannels).fill(0),
        new Array(image.numChannels).fill(0),
        new Array(image.numChannels).fill(0),
    ];

    const defaultMaps = ['', 'w', 'gm', 'rgb', 'mcgr', '-mgcr', '--mgcr', '---mgcr'];
    console.log(defaultMaps[exampleImage.numChannels])
    setColorMat(image, defaultMaps[exampleImage.numChannels]);
}


export function normalizeColorMat(colormat) {
    let maxTotal = 0;
    for (let arr of colormat) {
        let total = 0;
        for (let val of arr) {
            total += val;
        }
        maxTotal = Math.max(maxTotal, total);
    }

    for (let i = 0; i < colormat[0].length; i ++) {
        colormat[0][i] /= maxTotal;
        colormat[1][i] /= maxTotal;
        colormat[2][i] /= maxTotal;
    }
    return colormat;
}
