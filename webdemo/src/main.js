'use strict';

import L from "leaflet";
import "leaflet/dist/leaflet.css";

import { ImageHandler } from "./inference";

var Jimp = require('jimp');

function counter() {
  let seconds = 0;
  setInterval(() => {
    seconds += 1;
    document.getElementById('app').innerHTML = `<p>You have been here for ${seconds} seconds.</p>`;
  }, 1000);
}

counter();


function initMap() {
  var map = L.map('output-map').setView([51.505, -0.09], 13);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);
}
initMap();


console.log("line 30");
console.log(Jimp);

const imgHandler = new ImageHandler(
  document.getElementById("input-image"),
  document.getElementById("input-image-canvas"),
);

function inputImageChanged(e) {
  const file = e.target.files[0];
  imgHandler.processUserFile(file);
}

document.getElementById("image-file")
  .addEventListener("change", inputImageChanged);