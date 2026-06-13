import {setColorMat, setBounds} from './settings';

function openCommandLine() {
    const input = document.querySelector('#command-line input');
    input.value = "";
    input.focus();
    console.log('hi');
}

const commands = {
    help: () => display("Available commands: " + Object.keys(commands)),
    greetings: () => console.log("Hello"),
    colors: (arg) => setColorMat(arg),
    print: console.log,
    fisseq: () => {setBounds(5000); setColorMat('--mcgr')},
    bounds: (...args) => setBounds(...args),
};

function display(message) {
    const input = document.querySelector('#command-line input');
    input.value = "# " + message;
    input.focus();
    input.select();
    //input.classList.add('message');
}

function runCommand(command) {
    if (command.length == 0 || command.charAt(0) == '#') return;

    if (command.charAt(0) == ':') {
        command = command.slice(1)
    }

    let args = command.split(' ');
    command = args[0];
    args = args.slice(1);

    let matchingCommands = [];
    for (const [name, func] of Object.entries(commands)) {
        if (name.length < command.length) continue;

        if (name.length >= command.length && command == name.slice(0, command.length)) {
            matchingCommands.push(name);
        }
    }

    if (matchingCommands.length == 0) {
        display("No command matching '" + command + "' found");
    } else if (matchingCommands.length > 1) {
        display("Ambiguous command, '" + command + "' matches all of " + matchingCommands);
    } else {
        try {
            commands[matchingCommands[0]](...args);
        } catch (error) {
            display(error.toString());
            throw error;
        }
    }
}

export function onKeyDownCommands(event) {
    if (event.target.tagName == "INPUT") {
        if (event.target.parentNode.id == "command-line") {
            if (event.target.classList.contains('message')) {
                event.target.classList.remove('message');
                event.target.value = (event.key == ':') ? "" : ":";
                event.target.focus();
            }
            if (event.key == "Escape") {
                event.target.blur();
            }
            if (event.key == "Enter") {
                event.target.blur();
                runCommand(event.target.value);
            }
        } else if (event.key == ':') {
            openCommandLine();
        }
        return false;
    }
    if (event.key == ':') {
        openCommandLine();
    }
    return true;
}

