'use strict';

import { S2 } from "s2-geometry";

import { LABEL_MAP } from "./label_map";

async function userFileToBase64(file) {
    const fileReader = new FileReader();
    return new Promise((resolve, reject) => {
        fileReader.addEventListener("load", () => {
            resolve(fileReader.result);
        });
        fileReader.addEventListener("error", () => {
            reject(fileReader.error);
        });
        fileReader.readAsDataURL(file);
    });
}

function normalizeChannel(channelArray, mean, std) {
    for (let i = 0; i < channelArray.length; i++) {
        channelArray[i] = (channelArray[i] - mean) / std;
    }
}

function tensorFromData(data) {
    const [redArray, greenArray, blueArray] = [[], [], []];

    console.log(data.data.length);
    for (let i = 0; i < data.data.length; i += 4) {
        redArray.push(data.data[i] / 255.0);
        greenArray.push(data.data[i + 1] / 255.0);
        blueArray.push(data.data[i + 2] / 255.0);
    }
    
    // Normalize to imagenet mean and std
    normalizeChannel(redArray, 0.485, 0.229);
    normalizeChannel(greenArray, 0.456, 0.224);
    normalizeChannel(blueArray, 0.406, 0.225);

    // Concatenate RGB to get [3, 224, 224]
    const transposedData = redArray.concat(greenArray).concat(blueArray);

    console.log(ort);
    console.log(ort.Tensor);

    const inputTensor = new ort.Tensor("float32", transposedData, [1, 3, 224, 224]);
    return inputTensor;
}

export class ImageInferenceHandler {
    constructor(previewImgEl, canvasEl, leafletMap, statusDisplay) {
        this.previewImgEl = previewImgEl;
        this.canvasEl = canvasEl;
        this.leafletMap = leafletMap;
        this.statusDisplay = statusDisplay;

        previewImgEl.addEventListener("load", (e) => {
            var ctx = this.canvasEl.getContext("2d");
            ctx?.drawImage(e.target, 0, 0, this.canvasEl.width, this.canvasEl.height);
            const data = ctx?.getImageData(0, 0, this.canvasEl.width, this.canvasEl.height);
            console.log(this.canvasEl.width, this.canvasEl.height);

            this.runInferenceDemo(data);
        });
    }

    async processUserFile(file) {
        this.statusDisplay.log("Processing image...");
        const imgBase64 = await userFileToBase64(file);
        this.previewImgEl.src = imgBase64;
    }

    async runInferenceDemo(data) {
        this.statusDisplay.log("Running inference...");
        const labelsArray = await this.runInferenceMultilabel(data);

        if (labelsArray.length === 0) {
            this.statusDisplay.error("Error: no location predicted :(");
            return;
        }

        this.statusDisplay.log("Rendering output...");
        const predictedLatLng = this.labelsToLatLng(labelsArray);
        console.log(predictedLatLng);

        // Clear existing markers
        const map = this.leafletMap;
        map.eachLayer(function(layer) {
            if (layer instanceof L.Marker)
            {
                map.removeLayer(layer);
            }
        });

        // Draw the predicted location on the map
        L.marker([predictedLatLng.lat, predictedLatLng.lng]).addTo(map);
        map.flyTo([predictedLatLng.lat, predictedLatLng.lng], 9);

        this.statusDisplay.log("Location: " + predictedLatLng.lat + ", " + predictedLatLng.lng);
    }

    async runInferenceMultilabel(data) {
        const imageTensor = tensorFromData(data);
        // console.log(imageTensor);
        console.log(S2.latLngToKey(37.7749, -122.4194, 15));

        /*const modelResponse = await fetch(
            //"https://github.com/fyhuang/img2loc/releases/download/v1/s2cell_ml_efn_v2_s2_train1.onnx",
            "https://objects.githubusercontent.com/github-production-release-asset-2e65be/765955984/ce794da9-0b29-452c-aa17-ae8e5d95ddc5?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20240619%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240619T210826Z&X-Amz-Expires=300&X-Amz-Signature=bbf31719947c2d2f4a2d64a74a6338ff0dfaad779263eb1930a7b4f0f98d99d2&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=765955984&response-content-disposition=attachment%3B%20filename%3Ds2cell_ml_efn_v2_s2_train1.onnx&response-content-type=application%2Foctet-stream",
            //{ mode: "no-cors" }
        );
        console.log(modelResponse);
        if (modelResponse.status !== 200) {
            console.error("Failed to fetch model", modelResponse.status, modelResponse.statusText);
            return [];
        }

        const modelArrayBuffer = await modelResponse.arrayBuffer();
        console.log(modelArrayBuffer);

        const session = await ort.InferenceSession.create(
            modelArrayBuffer,
            { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
        );*/


        const session = await ort.InferenceSession.create(
            "https://media.githubusercontent.com/media/fyhuang/img2loc/main/exports/s2cell_ml_efn_v2_s2_train1.onnx",
            { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
        );

        const feeds = {};
        feeds[session.inputNames[0]] = imageTensor;

        const outputFeeds = await session.run(feeds);
        const outputTensor = outputFeeds[session.outputNames[0]];

        // Is the output already softmaxed?
        console.log(outputTensor);

        // Get top classes
        const labelsArray = [];
        const outputArray = await outputTensor.getData();
        for (let i = 0; i < outputArray.length; i++) {
            if (outputArray[i] >= 0.5) {
                labelsArray.push(i);
            }
        }

        console.log(labelsArray);
        return labelsArray;
    }

    labelsToLatLng(labelsArray) {
        // The best cell is the one that has the most ancestors in the prediction set
        const labelsSet = new Set(labelsArray);
        const labelToParents = new Map();
        for (const outputLabel of labelsArray) {
            labelToParents.set(outputLabel, 0);

            // Count number of parents
            const cellObj = LABEL_MAP.get(outputLabel);
            for (const ancestorLabel of cellObj.ancestors) {
                if (labelsSet.has(ancestorLabel)) {
                    labelToParents.set(
                        outputLabel,
                        labelToParents.get(outputLabel) + 1
                    );
                }
            }
        }

        // In the case of a tie, we pick the cell with the lowest level (to reflect uncertainty)
        const sortedLabels = Array.from(labelsArray).sort((a, b) => {
            const parentsA = labelToParents.get(a);
            const parentsB = labelToParents.get(b);
            if (parentsA === parentsB) {
                return LABEL_MAP.get(a).level - LABEL_MAP.get(b).level;
            }
            return parentsB - parentsA;
        });

        return {
            lat: LABEL_MAP.get(sortedLabels[0]).lat,
            lng: LABEL_MAP.get(sortedLabels[0]).lng
        };
    }
}