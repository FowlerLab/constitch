import {globalScreenBox, assignIds, drawMinimap, updateMinimap, resetScreenBox, updateActiveImages} from './display';
import {defaultBounds, defaultColorMat} from './colors';
import {exampleImage} from './config';


if (!('id' in exampleImage)) {
    assignIds(exampleImage, 0);
}

if (!('bounds' in exampleImage)) {
    defaultBounds(exampleImage);
}

if (!('colorMat' in exampleImage)) {
    defaultColorMat(exampleImage);
}

// SETUP

//setColorMat('--mcgr');
//setBounds(5000);
//setBounds('1%');
//updateColorMap();
drawMinimap(exampleImage, exampleImage);
resetScreenBox();
updateMinimap(globalScreenBox, exampleImage);
updateActiveImages(globalScreenBox, exampleImage.colorMat, exampleImage.bounds, exampleImage);
