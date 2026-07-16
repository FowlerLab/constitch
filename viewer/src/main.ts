import {globalScreenBox, assignIds, drawMinimap, updateMinimap, resetScreenBox, updateActiveImages} from './display';
import {setColorMat, setBounds} from './settings';
import {globalColorMat} from './colors';
import {info, exampleImage} from './config';


if (!('id' in exampleImage)) {
    assignIds(exampleImage, 0);
}

// SETUP

setColorMat('--mcgr');
setBounds(5000);
//setBounds('1%');
//updateColorMap();
drawMinimap(exampleImage, exampleImage);
resetScreenBox();
updateMinimap(globalScreenBox, exampleImage);
updateActiveImages(globalScreenBox, globalColorMat, exampleImage);
