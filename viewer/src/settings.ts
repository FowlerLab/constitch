import {onScreenChange, onSettingsChange} from './display';
import {globalColorMat, globalBounds} from './colors';
import {info} from './config';

/// Form functions

for (let elem of document.querySelectorAll('.channel0')) {
    for (let i = 1; i < 6; i ++) {
        const newelem = elem.cloneNode(true);
        newelem.classList.remove('channel0');
        newelem.classList.add('channel' + i);
        newelem.querySelectorAll('[data-ch' + i + 'class]').forEach(
                subelem => subelem.classList.add(subelem.dataset['ch' + i + 'class']));
        newelem.querySelectorAll('[name]').forEach(
                subelem => subelem.name = subelem.name.replaceAll('{channel}', i));
        elem.parentNode.appendChild(newelem);
    }

    elem.querySelectorAll('[data-ch0class]').forEach(
            subelem => subelem.classList.add(subelem.dataset['ch0class']));
    elem.querySelectorAll('[name]').forEach(
            subelem => subelem.name = subelem.name.replaceAll('{channel}', '0'));
}

document.querySelectorAll('.channels input[type="radio"]').forEach(elem => {
    elem.addEventListener('click', event => {
        elem.closest('.dropdown').classList.toggle('open');
        if (updateColorMap()) onFormUpdate();
    });
    //elem.addEventListener('onchange', event => {
        //console.log('changing');
    //});
})

document.querySelectorAll('.bounds input[type="range"]').forEach(elem => {
    elem.addEventListener('change', event => {
        if (updateBounds()) onFormUpdate();
    })
});

function parseNumber(str) {
    console.log('parsing', str);
    const lastLetter = str.charAt(str.length - 1).toLowerCase();
    console.log(lastLetter);
    if (lastLetter == 'k') {
        console.log(parseNumber(str.slice(0, -1)), str.slice(0, -1))
        return parseNumber(str.slice(0, -1)) * 1000;
    } else if (lastLetter == 'm') {
        return parseNumber(str.slice(0, -1)) * 1000_000;
    } else if (lastLetter == 'g') {
        return parseNumber(str.slice(0, -1)) * 1000_000_000;
    } else if (lastLetter == 't') {
        return parseNumber(str.slice(0, -1)) * 1000_000_000_000;
    }
    return Number(str);
}

function onFormUpdate() {
    onScreenChange();
}

function formBoundsUpdate(elem) {
    if (elem.classList.contains('max')) {
        const otherInput = elem.closest('.bounds').querySelector('.max')
        if (Number(otherInput.value) < Number(elem.value)) {
            otherInput.value = elem.value;
        }
    } else {
        const otherInput = elem.closest('.bounds').querySelector('.min')
        if (Number(otherInput.value) > Number(elem.value)) {
            otherInput.value = elem.value;
        }
    }
    onFormUpdate();
}

const colorKey = {
    r: [1, 0, 0],
    g: [0, 1, 0],
    b: [0, 0, 1],
    m: [1, 0, 1],
    c: [0, 1, 1],
    y: [1, 1, 0],
    w: [1, 1, 1],
    "-": [0, 0, 0],
}

function updateColorMap() {
    const newmat = [[], [], []];
    for (let i = 0; i < globalColorMat[0].length; i ++) {
        //const color = JSON.parse(document.querySelector('.colorselector.channel' + i + ' .selected').dataset.color);
        const colorCode = document.querySelector('input[name="ch' + i + 'color"]:checked').value
        const color = colorKey[colorCode] ?? [0, 0, 0];
        const scalar = 1;
        newmat[0].push(color[0] * scalar);
        newmat[1].push(color[1] * scalar);
        newmat[2].push(color[2] * scalar);
    }
    if (JSON.stringify(newmat) == JSON.stringify(globalColorMat)) {
        console.log('not changed');
        return false;
    }
    //normalizeColorMat(newmat);

    globalColorMat[0] = newmat[0];
    globalColorMat[1] = newmat[1];
    globalColorMat[2] = newmat[2];
    console.log(JSON.stringify(newmat))
    onSettingsChange();
    //lastUpdate = Date.now();
    //renderFlag = true;
    return true;
}

export function setColorMat(arg) {
    if (arg.length > globalColorMat[0].length) {
        throw new Error("Color string has length greater than the number of channels");
    }

    for (let i = 0; i < globalColorMat[0].length; i ++) {
        const char = (i >= arg.length) ? arg.charAt(arg.length - 1) : arg.charAt(i);
        //console.log('input[name="ch' + i + 'color"][value="' + char + '"]');
        document.querySelector('input[name="ch' + i + 'color"][value="' + char + '"]').checked = true;
    }

    if (updateColorMap()) onFormUpdate();
}

function formColorPicker(event) {
    event.target.classList.toggle('selected');
    event.target.closest('.colorselector').classList.toggle('open');
    if (event.target.closest('.channel-colorselectors').querySelector('.open') == null) {
        console.log('UPDATING');
        updateColorMap();
        onFormUpdate();
        console.log(globalColorMat);
    }
}

function updateBounds() {
    const newbounds = [];
    for (let i = 0; i < globalBounds.length; i ++) {
        newbounds.push([
            Number(document.querySelector('.bounds input[name="channel' + i + 'min"]').value),
            Number(document.querySelector('.bounds input[name="channel' + i + 'max"]').value),
        ])
    }

    if (JSON.stringify(newbounds) == JSON.stringify(globalBounds)) {
        console.log('not changed');
        return false;
    }
    console.log(JSON.stringify(newbounds));

    console.log('global', JSON.stringify(globalBounds));
    for (let i = 0; i < globalBounds.length; i ++) {
        globalBounds[i] = newbounds[i];
    }
    console.log('global', JSON.stringify(globalBounds));

    onSettingsChange();
    return true;
}

export function setBounds(...args) {
    console.log('args', JSON.stringify(args));
    const parsearg = (arg) => {
        console.log('parse', arg);
        if (Array.isArray(arg)) {
            return arg.map(parsearg);
        }

        if (typeof arg == "string") {
            console.log('hi');
            if (arg.includes('-')) {
                console.log('hi2');
                return arg.split('-').map(parsearg);
            }
            if (arg.at(-1) == '%') {
            console.log('hi3');
                console.log('ksjdflksdjf');
                return arg;
            }
            console.log('hi4');
            return parseNumber(arg);
        }

        return arg;
    }

    let bounds = parsearg(args);
    console.log('bounds', JSON.stringify(bounds));
    if (bounds.length == 1) {
    //if (!Array.isArray(bounds) || (bounds.length == 2 && globalBounds.length != 2)) {
        if (typeof bounds[0] == "string" && bounds.charAt(bounds.length - 1) == '%') {
            let percent = Number(bounds[i].slice(0, -1));
            if (percent > 50) percent = 100 - percent;
            bounds = [info['percentiles'][percent], info['percentiles'][100-percent]]
        }
        bounds = Array(globalBounds.length).fill(bounds[0]);
    }

    for (let i = 0; i < bounds.length; i ++) {
        if (!Array.isArray(bounds[i])) {
            if (typeof bounds[i] == "string" && bounds[i].at(-1) == '%') {
                let percent = Number(bounds[i].slice(0, -1));
                if (percent > 50) percent = 100 - percent;
                bounds[i] = [info['percentiles_separate'][percent][i], info['percentiles_separate'][100-percent][i]]
            } else {
                bounds[i] = [0, bounds[i]]
            }
        }
    }

    console.log(JSON.stringify(bounds));
    if (bounds.length != globalBounds.length) {
        throw new Error('The number of arguments must be 1 or the number of channels');
    }

    for (let i = 0; i < bounds.length; i ++) {
        document.querySelector('.bounds input[name="channel' + i + 'min"]').value = bounds[i][0];
        document.querySelector('.bounds input[name="channel' + i + 'max"]').value = bounds[i][1];
    }

    if (updateBounds()) onFormUpdate();
}
