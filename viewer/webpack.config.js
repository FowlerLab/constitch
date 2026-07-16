const path = require('path');
const webpack = require('webpack');

const HtmlWebpackPlugin = require('html-webpack-plugin');
const HtmlInlineScriptPlugin = require('html-inline-script-webpack-plugin');

module.exports = (env) => {
    if (env.production) {
        return {
            mode: "production",
            entry: {
                main: "./src/main.ts",
            },
            output: {
                path: path.resolve(__dirname, './build'),
                filename: "bundle.js" // <--- Will be compiled to this single file
            },
            resolve: {
                extensions: [".ts", ".tsx", ".js"],
            },
            plugins: [new webpack.NormalModuleReplacementPlugin(
                    /.\/config\.ts/,
                    './config.production.ts'
                ), new HtmlWebpackPlugin({
                    template: "src/index.html",
                }), new HtmlInlineScriptPlugin()],
        };
    } else {
        return {
            mode: "development",
            devtool: "inline-source-map",
            entry: {
                main: "./src/main.ts",
            },
            output: {
                path: path.resolve(__dirname, './build'),
                filename: "bundle.js" // <--- Will be compiled to this single file
            },
            resolve: {
                extensions: [".ts", ".tsx", ".js"],
            },
            plugins: [new HtmlWebpackPlugin({
                    filename: "test.html",
                    template: "src/index.html",
                }), new HtmlInlineScriptPlugin()],
        };
    }
};
