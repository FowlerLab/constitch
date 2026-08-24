
export let info = {
    'shape': [6, 1000, 1000],
    'numchannels': 6,
    'dtype': 'uint16',
    'percentiles': {0.1: 96, 1: 111, 99: 3132, 99.9: 8087},
    'percentiles_separate': {
        0.1: [  264,    83,   173,   127,   362,   421],
        1:   [  291,    99,   196,   144,   399,   460],
        99:  [ 2187,  1002,  3214,  3633,  5026,  3537],
        99.9:[ 3124,  1738,  6629,  8010, 13688,  7399],
    },
};

export let exampleImage = {
    box: [0, 0, 1000, 1000],
    dims: [125, 125],
    shape: [6, 125, 125],
    url: 'tmp_input_125.png',
    images: [{
        box: [0, 0, 1000, 1000],
        dims: [250, 250],
        shape: [6, 250, 250],
        url: 'tmp_input_250.png',
        images: [{
            box: [0, 0, 1000, 1000],
            dims: [500, 500],
            shape: [6, 500, 500],
            url: 'tmp_input_500.png',
            images: [{
                box: [0, 0, 1000, 1000],
                dims: [1000, 1000],
                shape: [6, 1000, 1000],
                url: 'tmp_input.png',
            }]
        }]
    }]
};

exampleImage = {
                box: [500, 500, 1000, 1000],
                dims: [1000, 1000],
                shape: [6, 1000, 1000],
                url: 'tmp_input.png',
            };

exampleImage = {
    box: [0, 0, 2000, 2000],
                dims: [1000, 1000],
                shape: [6, 1000, 1000],
    images: [{
            box: [0, 0, 1000, 1000],
            dims: [1000, 1000],
            shape: [6, 1000, 1000],
            url: 'tmp_input.png',
        }, {
            box: [1000, 500, 1000, 1000],
            dims: [1000, 1000],
            shape: [6, 1000, 1000],
            url: 'tmp_input.png',
        },
    ]};
